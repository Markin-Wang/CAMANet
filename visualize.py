import torch
import argparse
import numpy as np
import os
import random
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
from models.model import R2GenModel
from modules.utils import parse_args, auto_resume_helper, load_checkpoint
from modules.logger import create_logger
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from config_swin import get_config
from tqdm import tqdm
import json

def main():
    # parse arguments
    args, config = parse_args()
    print(args)
    # create tokenizer
    tokenizer = Tokenizer(args)

    # create data loader
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    # build model architecture
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")
    model = R2GenModel(args, tokenizer, logger, config)
    model.load_state_dict(torch.load(args.pretrained)['state_dict'])


    # if dist.get_rank() == 0:
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")
    logger.info(config.dump())

    metrics = compute_scores

    model = model.cuda()

    model.eval()
    data = []
    with torch.no_grad():
        records = {}
        val_gts, val_res = [], []
        for batch_idx, (images_id, images, reports_ids, reports_masks, labels) in enumerate(tqdm(test_dataloader)):
            vis_data = {}
            vis_data['id'] = images_id
            vis_data['img'] = images
            vis_data['labels'] = labels

            images, reports_ids, reports_masks, labels = images.cuda(), reports_ids.cuda(), \
                                                         reports_masks.cuda(), labels.cuda()
            #if images_id[0] != 'data/mimic_cxr/images/p10/p10402372/s51966612/8797515b-595dfac0-77013a06-226b52bd-65681bf2.jpg':
            #    continue
            #print('000', reports_ids, reports_ids.shape)


            output, attns  = model(images, mode='sample')
            vis_data['attn'] = attns
            #output, img_con_ls, txt_con_ls, img_cls, txt_cls, attn_weights = model(images, reports_ids, labels=labels, mode='train')
            # change to self.model.module for multi-gpu

            #torch.save(attn_weights, 'attn_weights.pth')
            #torch.save(reports_ids, 'id.pth')




            if args.n_gpu > 1:
                reports = model.module.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = model.module.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
            else:
                reports = model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())

            vis_data['pre'] = reports
            vis_data['gt'] = ground_truths
            vis_data['met'] = []


            for id, predict, gt in zip(images_id, reports, ground_truths):
                val_met = metrics({id: [gt]}, {id: [predict]})
                # if val_met['BLEU_4'] > 0.3:
                #     records[id] = {'predict': predict, 'ground truth': gt, 'met': val_met, 'label': label}
                vis_data['met'].append(val_met)
            data.append(vis_data)
            if batch_idx >3:
                break

    # f = open('mimic_prediction_our03.json', 'w', encoding='utf-8')
    # json.dump(records, f, indent=1)
    # f.close()
    torch.save(data, args.exp_name+'_vis_data.pth')









if __name__ == '__main__':
    main()
