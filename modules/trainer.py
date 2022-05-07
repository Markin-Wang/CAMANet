import os
from abc import abstractmethod

import time
import torch
import datetime
import pandas as pd
from numpy import inf
from tqdm import tqdm
from modules.utils import auto_resume_helper
try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

from modules.utils import get_grad_norm
import torch.distributed as dist

class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, lr_scheduler, args, logger, config):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.lr_scheduler = lr_scheduler

        # if config.AMP_OPT_LEVEL != "O0":
        #     model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK],
        #                                                   broadcast_buffers=False, find_unused_parameters=True)
        self.model = model
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        if args.addcls:
            self.cls_criterion = torch.nn.BCEWithLogitsLoss()
            self.cls_w = args.cls_w
        self.attn_cam = args.attn_cam
        if args.attn_cam:
            self.mse_criterion = torch.nn.MSELoss()
            self.mse_w = args.mse_w

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.logger = logger

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.mnt_test_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = os.path.join(args.save_dir, args.exp_name)
        self.best_epoch = 0

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume:
            self._resume_checkpoint(self.checkpoint_dir)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        start = time.time()
        end = time.time()
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            # print logged informations to the screen

            for key, value in log.items():
                self.logger.info('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    cur_metric = log['val_BLEU_4'] + 0.5 * log['val_METEOR'] + 0.125 * log['val_BLEU_1']
                    improved = (self.mnt_mode == 'min' and cur_metric <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and cur_metric > self.mnt_best)
                except KeyError:
                    self.logger.info("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = cur_metric
                    not_improved_count = 0
                    best = True
                    self.best_epoch = epoch
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break
            self.logger.info('current best model in: {}'.format(self.best_epoch))

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
        epoch_time = time.time() - start
        self.logger.info(f"EPOCH {self.epochs} training takes {datetime.timedelta(seconds=int(epoch_time))}")
        self._print_best()
        self._print_best_to_file()

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'
        record_dir = os.path.join(self.args.record_dir, self.args.dataset_name)
        if not os.path.exists(record_dir):
            os.makedirs(record_dir)
        record_path = os.path.join(record_dir, self.args.exp_name+'.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
        record_table = record_table.append(self.best_recorder['test'], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'monitor_test_best': self.mnt_test_best,
            'best_epoch':self.best_epoch,
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_file = auto_resume_helper(resume_path)
        #resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_file))
        checkpoint = torch.load(resume_file)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.mnt_test_best = checkpoint['monitor_test_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_epoch = checkpoint['best_epoch']
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        cur_metric = log['val_BLEU_4'] + 0.5 * log['val_METEOR'] + 0.125 * log['val_BLEU_1']
        improved_val = (self.mnt_mode == 'min' and cur_metric <= self.mnt_best) or \
                       (self.mnt_mode == 'max' and cur_metric > self.mnt_best)
        if improved_val:
            self.best_recorder['val'].update(log)

        cur_metric = log['test_BLEU_4'] + 0.5 * log['test_METEOR'] + 0.125 * log['test_BLEU_1']

        improved_test = (self.mnt_mode == 'min' and cur_metric <= self.mnt_test_best) or \
                        (self.mnt_mode == 'max' and cur_metric > self.mnt_test_best)
        if improved_test:
            self.best_recorder['test'].update(log)
            self.mnt_test_best = cur_metric

    def _print_best(self):
        print('exp_name:', self.args.exp_name)
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, tokenizer,
                 train_dataloader, val_dataloader, test_dataloader, writer, logger, config):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, lr_scheduler, args, logger, config)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.writer = writer
        self.config = config
        self.tokenizer = tokenizer
        self.addcls = args.addcls
        self.early_exit = args.early_exit

    def _train_epoch(self, epoch):

        ce_losses = 0
        img_cls_losses = 0
        mse_losses = 0
        val_ce_losses = 0
        val_img_cls_losses = 0
        val_mse_losses = 0
        num_steps = len(self.train_dataloader)
        self.model.train()
        cur_lr = [param_group['lr'] for param_group in self.optimizer.param_groups]
        with tqdm(desc='Epoch %d - train, lr:(%.5f,%.5f)' % (epoch, cur_lr[0], cur_lr[1]),
                  unit='it', total=len(self.train_dataloader)) as pbar:
            for batch_idx, (images_id, images, reports_ids, reports_masks, labels) in enumerate(self.train_dataloader):
                images, reports_ids, reports_masks, labels = images.to(self.device, non_blocking=True), \
                                                     reports_ids.to(self.device, non_blocking=True), \
                                                     reports_masks.to(self.device, non_blocking=True), \
                                                     labels.to(self.device, non_blocking = True)
                logits, total_attn = None, None
                if self.addcls:
                    output, logits, cam, fore_map, total_attn = self.model(images, reports_ids, labels, mode='train')
                else:
                    output = self.model(images, reports_ids, mode='train')
                loss = self.criterion(output, reports_ids, reports_masks)

                ce_losses += loss.item()
                if logits is not None:
                    img_cls_loss = self.cls_criterion(logits, labels)
                    loss = loss + self.cls_w * img_cls_loss
                    img_cls_losses += img_cls_loss.item()

                if total_attn is not None:
                    mse_loss = self.mse_criterion(fore_map,total_attn)
                    loss = loss + self.mse_w * mse_loss
                    mse_losses += mse_loss.item()


                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
                self.lr_scheduler.step_update((epoch-1) * num_steps + batch_idx)
                # self.lr_scheduler.step_update((epoch) * num_steps + batch_idx)
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                cur_lr = [param_group['lr'] for param_group in self.optimizer.param_groups]
                pbar.set_postfix(ce_ls=ce_losses / (batch_idx + 1), cls_ls=img_cls_losses / (batch_idx + 1),
                                 mse_ls = mse_losses / (batch_idx + 1), mem = f'mem {memory_used:.0f}MB')
                pbar.update()
                if self.early_exit and batch_idx>100:
                    torch.save(self.model.records, 'cam_records_fblrelu.pth')
                    exit()
            log = {'ce_loss': ce_losses / len(self.train_dataloader)}
        self.writer.add_scalar('data/ce_loss', ce_losses, epoch)
        self.writer.add_scalar('data/cls_loss', img_cls_losses, epoch)
        self.writer.add_scalar('data/mse_loss', mse_losses, epoch)

        self.model.eval()
        with tqdm(desc='Epoch %d - val' % epoch, unit='it', total=len(self.val_dataloader)) as pbar:
            with torch.no_grad():
                val_gts, val_res = [], []
                for batch_idx, (images_id, images, reports_ids, reports_masks,labels) in enumerate(self.val_dataloader):
                    images, reports_ids, reports_masks,labels = images.to(self.device,non_blocking=True), \
                                                         reports_ids.to(self.device,non_blocking=True), \
                                                         reports_masks.to(self.device, non_blocking=True), \
                                                         labels.to(self.device, non_blocking = True)
                    total_attn = None
                    if self.addcls:
                        out, logits, cam, fore_map, total_attn = self.model(images, reports_ids, labels, mode='train')
                        val_img_cls_loss = self.cls_criterion(logits,labels)
                        val_img_cls_losses += val_img_cls_loss.item()
                    else:
                        out = self.model(images, reports_ids, mode='train')

                    output = self.model(images, mode='sample')

                    if total_attn is not None:
                        mse_loss = self.mse_criterion(fore_map, total_attn)
                        val_mse_losses += mse_loss.item()
                    loss = self.criterion(out, reports_ids, reports_masks)
                    val_ce_losses += loss.item()
                    reports = self.tokenizer.decode_batch(output.cpu().numpy())
                    ground_truths = self.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                    val_res.extend(reports)
                    val_gts.extend(ground_truths)
                    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                    pbar.set_postfix(ce_ls=val_ce_losses / (batch_idx + 1), cls_ls=val_img_cls_losses / (batch_idx + 1),
                                     mse_ls = val_mse_losses / (batch_idx + 1), mem=f'mem {memory_used:.0f}MB')
                    pbar.update()
                val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                           {i: [re] for i, re in enumerate(val_res)})
                log.update(**{'val_' + k: v for k, v in val_met.items()})
                log.update({'val_loss': val_ce_losses / len(self.val_dataloader)})
                self.writer.add_scalar('data/val_bleu1', val_met['BLEU_1'], epoch)
                self.writer.add_scalar('data/val_bleu2', val_met['BLEU_2'], epoch)
                self.writer.add_scalar('data/val_bleu3', val_met['BLEU_3'], epoch)
                self.writer.add_scalar('data/val_bleu4', val_met['BLEU_4'], epoch)
                self.writer.add_scalar('data/val_meteor', val_met['METEOR'], epoch)
                self.writer.add_scalar('data/val_rouge-l', val_met['ROUGE_L'], epoch)
        self.writer.add_scalar('data/val_ce_loss', val_ce_losses, epoch)
        self.writer.add_scalar('data/val_cls_loss', val_img_cls_losses, epoch)
        self.writer.add_scalar('data/val_mse_loss', val_mse_losses, epoch)


        self.model.eval()
        with tqdm(desc='Epoch %d - test' % epoch, unit='it', total=len(self.test_dataloader)) as pbar:
            with torch.no_grad():
                test_gts, test_res = [], []
                for batch_idx, (images_id, images, reports_ids, reports_masks,labels) in enumerate(self.test_dataloader):
                    images, reports_ids, reports_masks, labels = images.to(self.device,non_blocking=True), \
                                                         reports_ids.to(self.device,non_blocking=True), \
                                                         reports_masks.to(self.device, non_blocking=True), \
                                                         labels.cuda(self.device, non_blocking=True)
                    #out = self.model(images, reports_ids, mode='train')
                    #loss = self.criterion(out, reports_ids, reports_masks)
                    output = self.model(images,  mode='sample')
                    reports = self.tokenizer.decode_batch(output.cpu().numpy())
                    ground_truths = self.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                    test_res.extend(reports)
                    test_gts.extend(ground_truths)
                    pbar.update()
                test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                            {i: [re] for i, re in enumerate(test_res)})
                log.update(**{'test_' + k: v for k, v in test_met.items()})
                self.writer.add_scalar('data/test_bleu1', test_met['BLEU_1'], epoch)
                self.writer.add_scalar('data/test_bleu2', test_met['BLEU_2'], epoch)
                self.writer.add_scalar('data/test_bleu3', test_met['BLEU_3'], epoch)
                self.writer.add_scalar('data/test_bleu4', test_met['BLEU_4'], epoch)
                self.writer.add_scalar('data/test_meteor', test_met['METEOR'], epoch)
                self.writer.add_scalar('data/test_rouge-l', test_met['ROUGE_L'], epoch)

        # add loss
        return log
