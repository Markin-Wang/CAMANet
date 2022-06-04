from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta
from modules.datasets import ChexPert
from torch.utils.data import DataLoader
from modules.dataloaders import R2DataLoader
from modules.tokenizers import Tokenizer
import json
from sklearn import metrics
from torchvision import transforms
import PIL
from modules.balanced_sampler import MultilabelBalancedRandomSampler

import torch
import torch.distributed as dist
from modules.optimizers import build_optimizer_cls, build_lr_scheduler

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# from apex import amp
# from apex.parallel import DistributedDataParallel as DDP

from modules.utils import parse_args
from models.classifier import Classifier
from easydict import EasyDict as edict
from timm.utils import accuracy, AverageMeter
from swintrans.utils import load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm, auto_resume_helper, \
    reduce_tensor
import datetime
import torch.nn.functional as F
import time

logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args, config, logger=None):
    # Prepare model

    if args.dataset_name == 'chexpert':
        num_classes = 14
    else:
        num_classes = 14

    model = Classifier(args, logger=logger, config=config, n_classes=num_classes)
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, writer=None):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        outputs = model(samples)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = criterion(outputs, targets)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    writer.add_scalar('data/train_loss', loss_meter.avg)
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def calculate_metricx(preds, targets):
    auclist = []
    for i in range(preds[0].shape[-1]):
        fpr, tpr, thresholds = metrics.roc_curve(targets[:, i], preds[:, i], pos_label=1)
        auclist.append(metrics.auc(fpr, tpr))
    pred_labels = preds > 0.5
    confusion_matrix = metrics.multilabel_confusion_matrix(y_true=targets, y_pred=pred_labels)
    return np.array([x for x in auclist if not np.isnan(x)]), confusion_matrix


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.BCEWithLogitsLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    auc_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    predlist = []
    true_list = []

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)

        # measure accuracy and record loss

        logits = F.torch.sigmoid(output)
        loss = criterion(logits, target)
        if idx == 0:
            predlist = logits.cpu().numpy()
            true_list = target.cpu().numpy()
        else:
            predlist = np.append(predlist, logits.cpu().numpy(), axis=0)
            true_list = np.append(true_list, target.cpu().numpy(), axis=0)
        pred_labels = logits.ge(0.5)
        acc = (target == pred_labels).float().sum() / (pred_labels.shape[-1] * pred_labels.shape[-2])
        # loss = reduce_tensor(loss)
        # acc = reduce_tensor(acc)
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                # f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    auc, confusion_matrix = calculate_metricx(predlist, true_list)
    return acc1_meter.avg, auc, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


def train(args, config, model):
    """ Train the model """
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("logs", args.exp_name))

    # args.batch_size = args.batch_size // args.gradient_accumulation_steps

    if args.dataset_name == 'chexpert':
        with open('./data/CheXpert-v1.0-small/example.json') as f:
            cfg = edict(json.load(f))
            train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomApply([
                    transforms.RandomRotation(10, interpolation=transforms.InterpolationMode.BICUBIC),
                    # transforms.RandomAffine(0, shear=10, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.RandomAffine(0, scale=(0.8, 1.2),
                                            interpolation=transforms.InterpolationMode.BICUBIC)
                ]),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomPerspective(distortion_scale=0.2),
                # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
                transforms.ToTensor(),
                # transforms.RandomErasing(scale=(0.02, 0.16), ratio=(0.3, 1.6)),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
            train_set = ChexPert(cfg.train_csv, cfg, mode='train', transform=train_transform)
            train_loader = DataLoader(
                train_set,
                batch_size=args.batch_size, num_workers=args.num_workers,
                drop_last=True, sampler=MultilabelBalancedRandomSampler(np.array(train_set._labels)))
            val_loader = DataLoader(
                ChexPert(cfg.dev_csv, cfg, mode='dev', transform=test_transform),
                batch_size=args.batch_size, num_workers=args.num_workers,
                drop_last=False, shuffle=False)
            test_loader = None
    else:
        tokenizer = Tokenizer(args)
        train_loader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
        val_loader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
        test_loader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    if args.finetune:
        state_dict = torch.load(args.pretrained)['model']
        logger.info(state_dict.keys())
        state_dict.pop('head.weight')
        state_dict.pop('head.bias')
        model.load_state_dict(state_dict, strict=False)

    # Prepare optimizer and scheduler
    # optimizer = torch.optim.SGD(model.parameters(),
    #                             lr=args.learning_rate,
    #                             momentum=0.9,
    #                             weight_decay=args.weight_decay)
    t_total = 1
    # if args.decay_type == "cosine":
    #     scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    # else:
    #     scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    optimizer = build_optimizer_cls(args, model)
    scheduler = build_lr_scheduler(config, optimizer, len(train_loader))

    # if args.fp16:
    #     model, optimizer = amp.initialize(models=model,
    #                                       optimizers=optimizer,
    #                                       opt_level=args.fp16_opt_level)
    #     amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    # Distributed training
    # if args.local_rank != -1:
    #     model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization epoch = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.batch_size)
    # logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
    #             args.train_batch_size * args.gradient_accumulation_steps * (
    #                 torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    # logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, max_accuracy = 0, 0.0
    criterion = torch.nn.BCEWithLogitsLoss()
    max_auc, max_auc_test = 0, 0
    start_time = time.time()
    best_epoch, best_epoch_test = 0, 0
    for epoch in range(args.epochs):
        # data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, train_loader, optimizer, epoch, None, scheduler, writer)

        acc1, auc, loss = validate(config, val_loader, model)
        auc_score = auc.mean()
        writer.add_scalar('data/val_loss', loss)
        writer.add_scalar('data/val_acc', acc1)
        writer.add_scalar('data/val_auc_score', auc_score)
        writer.add_text('data/val_auc', str(auc))
        if auc_score > max_auc:
            max_auc = auc_score
            best_epoch = epoch
            save_checkpoint(config, args, epoch, model, max_auc, optimizer, scheduler, logger)
        if test_loader:
            acc1_test, auc_test, loss_test = validate(config, test_loader, model)
            auc_score_test = auc_test.mean()
            writer.add_scalar('data/test_loss', loss_test)
            writer.add_scalar('data/test_acc', acc1_test)
            writer.add_scalar('data/test_auc_score', auc_score_test)
            writer.add_text('data/test_auc', str(auc_test))
            if auc_score_test > max_auc_test:
                max_auc_test = auc_score_test
                best_epoch_test = epoch
                # save_checkpoint(config, args, epoch, model, max_auc, optimizer, scheduler, logger)

        logger.info('Auc for all classes: ' + ', '.join([str(round(x.item(), 5)) for x in auc]))
        logger.info(f' * Auc@1 {auc.mean():.3f}')
        logger.info(f' * Acc@1 {acc1:.3f} ')
        logger.info(f'Best model in epoch: {best_epoch}')

    logger.info(f"Auc of the network on the {len(val_loader)} test images: {max_auc:.5f}%")
    logger.info(f'Max auc: {max_auc:.5f}%')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

    if args.local_rank in [-1, 0]:
        writer.close()
    # logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
    logger.info(f"exp name:{args.exp_name}")


def main():
    args, config = parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)
    torch.backends.cudnn.benchmark = True

    # Model & Tokenizer Setup
    args, model = setup(args, config, logger)

    # Training
    train(args, config, model)


if __name__ == "__main__":
    main()