import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from .datasets import IuxrayMultiImageDataset, MimiccxrSingleImageDataset,IuxrayMultiImageClsDataset,MimiccxrSingleImageClsDataset
from torch.utils.data import DistributedSampler
import torch.distributed as dist
from .balanced_sampler import MultilabelBalancedRandomSampler

class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle, vis = False):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split
        self.drop_last = True if split =='train' else False
        self.vis = vis

        if split == 'train':
            if args.randaug:
                print('Random applied transformation is utilized for ' + split +' dataset.')
                self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomApply([
                    transforms.RandomRotation(10, interpolation=transforms.InterpolationMode.BICUBIC),
                    #transforms.RandomAffine(0, shear=10, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.RandomAffine(0, scale=(0.8, 1.2),
                                            interpolation=transforms.InterpolationMode.BICUBIC)
                ]),
                #transforms.RandomHorizontalFlip(),
                # transforms.RandomPerspective(distortion_scale=0.2),
                # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
                transforms.ToTensor(),
                # transforms.RandomErasing(scale=(0.02, 0.16), ratio=(0.3, 1.6)),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        if self.dataset_name == 'iu_xray' and not args.cls:
            self.dataset = IuxrayMultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        elif self.dataset_name == 'iu_xray' and args.cls:
            self.dataset = IuxrayMultiImageClsDataset(self.args, self.split, transform=self.transform, vis = self.vis)
        elif self.dataset_name.startswith('mimic') and not args.cls:
            self.dataset = MimiccxrSingleImageDataset(self.args,  self.tokenizer, self.split, transform=self.transform)
        elif self.dataset_name.startswith('mimic') and args.cls:
            self.dataset = MimiccxrSingleImageClsDataset(self.args, self.split, transform=self.transform, vis = self.vis)

        if args.balanced:
            if split == 'train' and not self.vis:
                print('Balanced sampler is established for ' + split +' dataset.')
                self.sampler = MultilabelBalancedRandomSampler(np.array(self.dataset._labels))
                self.init_kwargs = {
                    'dataset': self.dataset,
                    'batch_size': self.batch_size,
                    'sampler': self.sampler,
                    'num_workers': self.num_workers,
                    'pin_memory': True,
                    'drop_last': self.drop_last,
                    #'collate_fn': self.collate_fn,
                    'prefetch_factor': self.batch_size // self.num_workers * 2
                }
            else:
                self.init_kwargs = {
                    'dataset': self.dataset,
                    # 'sampler': self.sampler,
                    'batch_size': self.batch_size,
                    'shuffle': shuffle,
                    'num_workers': self.num_workers,
                    'pin_memory': True,
                    'drop_last': self.drop_last,
                    #'collate_fn': self.collate_fn,
                    'prefetch_factor': self.batch_size // self.num_workers * 2
                }

        else:
            self.init_kwargs = {
                'dataset': self.dataset,
                # 'sampler': self.sampler,
                'batch_size': self.batch_size,
                'shuffle':shuffle,
                'collate_fn': self.collate_fn,
                'num_workers': self.num_workers,
                'pin_memory': True,
                'drop_last': self.drop_last,
                'prefetch_factor': self.batch_size // self.num_workers * 2
            }


        # num_tasks = dist.get_world_size()
        # global_rank = dist.get_rank()
        #
        # self.sampler = DistributedSampler(self.dataset, num_replicas=num_tasks,
        #                                   rank=global_rank, shuffle=self.shuffle)

        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        images_id, images, reports_ids, reports_masks, seq_lengths, labels = zip(*data)
        images = torch.stack(images, 0)
        max_seq_length = max(seq_lengths)
        labels = np.array(labels)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        return images_id, images, torch.LongTensor(targets), torch.FloatTensor(targets_masks), torch.FloatTensor(labels)

