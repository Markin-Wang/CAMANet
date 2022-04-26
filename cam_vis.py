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
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
from torchvision import transforms
import PIL
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from modules.utils import parse_args
from models.classifier import Classifier
from easydict import EasyDict as edict
from timm.utils import accuracy, AverageMeter
from swintrans.utils import load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor
import datetime
import torch.nn.functional as F
import time
from torchcam.methods import SmoothGradCAMpp, XGradCAM, CAM


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

    model = Classifier(args, logger = logger, config = config, n_classes=num_classes)
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def calculate_metricx(preds, targets):
    auclist = []
    for i in range(preds[0].shape[-1]):
        fpr, tpr, thresholds = metrics.roc_curve(targets[:, i], preds[:,i], pos_label=1)
        auclist.append(metrics.auc(fpr,tpr))
    pred_labels = preds > 0.5
    confusion_matrix = metrics.multilabel_confusion_matrix(y_true = targets, y_pred = pred_labels)
    return torch.from_numpy(np.array(auclist)), confusion_matrix

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
        loss = criterion(output, target)
        logits = F.torch.sigmoid(output)
        if idx == 0:
            predlist = logits.cpu().numpy()
            true_list = target.cpu().numpy()
        else:
            predlist = np.append(predlist, logits.cpu().numpy(), axis=0)
            true_list = np.append(true_list, target.cpu().numpy(), axis=0)
        pred_labels = logits.ge(0.5)
        acc = (target == pred_labels).float().sum() / (pred_labels.shape[-1]*pred_labels.shape[-2])
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
                #f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
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
    # os.makedirs(args.save_dir, exist_ok=True)
    # writer = SummaryWriter(log_dir=os.path.join("logs", args.exp_name))

    #args.batch_size = args.batch_size // args.gradient_accumulation_steps

    if args.dataset_name == 'chexpert':
        with open('./example.json') as f:
            cfg = edict(json.load(f))
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                # transforms.RandomCrop(224),
                # transforms.RandomApply([
                #     transforms.RandomRotation(15, resample=PIL.Image.BICUBIC),
                #     transforms.RandomAffine(0, translate=(
                #         0.2, 0.2), resample=PIL.Image.BICUBIC),
                #     transforms.RandomAffine(0, shear=20, resample=PIL.Image.BICUBIC),
                #     transforms.RandomAffine(0, scale=(0.8, 1.2),
                #                             resample=PIL.Image.BICUBIC)
                # ]),
                # transforms.RandomHorizontalFlip(),
                #transforms.RandomPerspective(distortion_scale=0.2),
                #transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
                transforms.ToTensor(),
                #transforms.RandomErasing(scale=(0.02, 0.16), ratio=(0.3, 1.6)),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
            train_loader = DataLoader(
                ChexPert(cfg.train_csv, cfg, mode='heatmap', transform=train_transform),
                batch_size=args.batch_size, num_workers=args.num_workers,
                drop_last=True, shuffle=True)
            val_loader = DataLoader(
                ChexPert(cfg.dev_csv, cfg, mode='heatmap',transform=test_transform),
                batch_size=args.batch_size, num_workers=args.num_workers,
                drop_last=False, shuffle=False)
            test_loader = val_loader
    else:
        tokenizer = Tokenizer(args)
        train_loader = R2DataLoader(args, tokenizer, split='train', shuffle=True, vis = True)
        val_loader = R2DataLoader(args, tokenizer, split='val', shuffle=False, vis = True)
        test_loader = R2DataLoader(args, tokenizer, split='test', shuffle=False, vis=True)


    logger.info("  Instantaneous batch size per GPU = %d", args.batch_size)


    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    global_step, max_accuracy = 0, 0.0
    criterion = torch.nn.BCEWithLogitsLoss()
    max_auc = 0
    start_time = time.time()
    best_epoch = 0
    cam_vis = True
    checkpoint = torch.load('pretrained_models/iu_finetune/iu_dense121_1e-5_50x_am_wd5e-5_wu0_1e-4_dc50_08_sd9223_ep5_bs32_n14_ft.pth')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    cam = []
    cam_extractor = CAM(model)
    to_img = transforms.ToPILImage()
    imgs = []
    #with torch.no_grad():
    count = 0
    for idx, (images, paths, target) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        print(idx)

        # compute output
        for image,path, tar in zip(images, paths, target):
            cur_cam = []
            out = model(image.unsqueeze(0))
            labels = torch.where(torch.sigmoid(out.squeeze(0))>0.5)[0]
            if len(labels)>0:
                count+=1
                for ids in labels:
                    #print(111111111111,idx)
                    activation_map = cam_extractor(ids.item(), out)
                    cur_cam.append({'path':path, 'map':activation_map, 'label':ids.item()})
                imgs.append({'image':image.detach().cpu(),'gt':tar.detach().cpu().numpy()})
                cam.append(cur_cam)
        if idx > 20:
            break

        # output = model(images)
        # #output = torch.sigmoid(output)
        # preds = [torch.where(F.sigmoid(o).flatten() > 0.5)[0] for o in output]
        # preds = torch.cat(preds)

        # cam_batch = cam_extractor(preds, output)



        # measure accuracy and record loss
        # loss = criterion(output, target)
        # logits = F.torch.sigmoid(output)
        # if idx == 0:
        #     predlist = logits.cpu().numpy()
        #     true_list = target.cpu().numpy()
        # else:
        #     predlist = np.append(predlist, logits.cpu().numpy(), axis=0)
        #     true_list = np.append(true_list, target.cpu().numpy(), axis=0)
        # pred_labels = logits.ge(0.5)
        # acc = (target == pred_labels).float().sum() / (pred_labels.shape[-1] * pred_labels.shape[-2])
        # loss = reduce_tensor(loss)
        # acc = reduce_tensor(acc)
        # loss_meter.update(loss.item(), target.size(0))
        # acc1_meter.update(acc.item(), target.size(0))


        # if idx % config.PRINT_FREQ == 0:
        #     memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        #     logger.info(
        #         f'Test: [{idx}/{len(test_loader)}]\t'
        #         f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #         f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
        #         f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
        #         # f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
        #         f'Mem {memory_used:.0f}MB')

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    denorm = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    image = imgs[0]['image']

    map1 = cam[3][0]['map'][0].squeeze(0)
    print(33333, map1.shape, count)
    # map2 = cam[3][1]['map'][0].squeeze(0)

    image = denorm(image)
    result1 = overlay_mask(transforms.functional.to_pil_image(image), transforms.functional.to_pil_image(map1, mode='F'), alpha=0.5)
    # result2 = overlay_mask(transforms.functional.to_pil_image(image), transforms.functional.to_pil_image(map2, mode='F'),
    #                       alpha=0.5)
    fig = plt.figure(figsize=(16,8))
    fig.add_subplot(1,2,1)
    plt.imshow(result1)
    plt.tight_layout()
    # fig.add_subplot(1, 2, 2)
    # plt.imshow(result2)
    # plt.tight_layout()
    plt.show()
    torch.save(imgs, 'chexpert_images.pth')
    torch.save(cam, 'chexpert_maps.pth')



        # auc, confusion_matrix = calculate_metricx(predlist, true_list)

    # auc_score = auc.mean()
    # writer.add_scalar('data/test_loss', loss)
    # writer.add_scalar('data/auc_score', auc_score)
    # writer.add_text('data/auc', str(auc))
    #
    # logger.info('Auc for all classes: '+', '.join([str(round(x.item(),5)) for x in auc]))
    # logger.info(f' * Auc@1 {auc.mean():.3f}')
    # logger.info(f'Best model in epoch: {best_epoch}')
    #
    # logger.info(f"Auc of the network on the {len(val_loader)} test images: {max_auc:.5f}%")
    # logger.info(f'Max auc: {max_auc:.5f}%')
    # writer.close()
    # #logger.info("Best Accuracy: \t%f" % best_acc)
    # logger.info("End Training!")


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

    # Model & Tokenizer Setup
    args, model = setup(args, config, logger)

    # Training
    train(args, config, model)


if __name__ == "__main__":
    main()