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


def main():
    # parse arguments
    args, config = parse_args()
    print(args)
    # if config.AMP_OPT_LEVEL != "O0":
    #     assert amp is not None, "amp not installed!"

    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

    # if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    #     rank = int(os.environ["RANK"])
    #     world_size = int(os.environ['WORLD_SIZE'])
    #     print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    # else:
    #     rank = -1
    #     world_size = -1

    # torch.cuda.set_device(config.LOCAL_RANK)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    # torch.distributed.barrier()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(config.OUTPUT, exist_ok=True)
    #logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")
    logger = create_logger(output_dir=config.OUTPUT,  name=f"{config.MODEL.NAME}")

    #if dist.get_rank() == 0:
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")
    logger.info(config.dump())


    # create tokenizer
    tokenizer = Tokenizer(args)

    # create data loader
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    # build model architecture
    model = R2GenModel(args, tokenizer, logger, config)

    resume = False
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(config, optimizer, len(train_dataloader))


    if args.finetune and not args.resume:
        state_dict = torch.load(args.pretrained)['model']
        logger.info(state_dict.keys())
        state_dict.pop('head.weight')
        state_dict.pop('head.bias')
        model.load_state_dict(state_dict, strict=False)
        logger.info(f'loading pretrained model {args.pretrained}, ignoring auto resume')

    model.cuda()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")
    print('SWR2Gen Transformer Training')

    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores

    # build optimizer, learning rate scheduler

    # build trainer and start to train
    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, tokenizer,
                      train_dataloader, val_dataloader, test_dataloader, writer, logger, config)
    trainer.train()
    writer.close()


if __name__ == '__main__':
    main()
