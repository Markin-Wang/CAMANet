import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset

import torchvision.transforms as tfs
import cv2
import pandas as pd
from .utils import GetTransforms, transform
import numpy as np


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class ChexPert(Dataset):
    def __init__(self, label_path, cfg, mode='train'):
        self.cfg = cfg
        self._label_header = None
        self._image_paths = []
        self._labels = []
        self._mode = mode
        self.dict = [{'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
                     {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}, ]
        self.transform = None
        with open(label_path) as f:
            header = f.readline().strip('\n').split(',')
            self._label_header = [
                header[7],
                header[10],
                header[11],
                header[13],
                header[15]]
            for line in f:
                labels = []
                fields = line.strip('\n').split(',')
                #print('111', fields)
                image_path = fields[0]
                flg_enhance = False
                for index, value in enumerate(fields[5:]):
                    if index == 5 or index == 8:
                        labels.append(self.dict[1].get(value))
                        if self.dict[1].get(
                                value) == '1' and \
                                self.cfg.enhance_index.count(index) > 0:
                            flg_enhance = True
                    elif index == 2 or index == 6 or index == 10:
                        labels.append(self.dict[0].get(value))
                        if self.dict[0].get(
                                value) == '1' and \
                                self.cfg.enhance_index.count(index) > 0:
                            flg_enhance = True
                    # else:
                    #     labels.append(self.dict[1].get(value))
                    #     if self.dict[1].get(
                    #             value) == '1' and \
                    #             self.cfg.enhance_index.count(index) > 0:
                    #         flg_enhance = True
                # labels = ([self.dict.get(n, n) for n in fields[5:]])
                labels = list(map(int, labels))
                image_path = os.path.join('./data', image_path)
                self._image_paths.append(image_path)
                assert os.path.exists(image_path), image_path
                self._labels.append(labels)
                if flg_enhance and self._mode == 'train':
                    for i in range(self.cfg.enhance_times):
                        self._image_paths.append(image_path)
                        self._labels.append(labels)
        self._num_image = len(self._image_paths)

    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        image = cv2.imread(self._image_paths[idx], 0)
        image = Image.fromarray(image)
        if self._mode == 'train':
            image = GetTransforms(image, type=self.cfg.use_transforms_type)
        image = np.array(image)
        image = transform(image, self.cfg)
        labels = np.array(self._labels[idx]).astype(np.float32)

        path = self._image_paths[idx]

        if self._mode == 'train' or self._mode == 'dev':
            return (image, labels)
        elif self._mode == 'test':
            return (image, path)
        elif self._mode == 'heatmap':
            return (image, path, labels)
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))

#
# class CheXpertDataSet(Dataset):
#     def __init__(self, image_list_file, transform=None, policy="ones"):
#         """
#         image_list_file: path to the file containing images with corresponding labels.
#         transform: optional transform to be applied on a sample.
#         Upolicy: name the policy with regard to the uncertain labels
#         """
#         image_names = []
#         labels = []
#         self.dict = [{'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
#                      {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}, ]
#
#         with open(image_list_file, "r") as f:
#             csvReader = csv.reader(f)
#             next(csvReader, None)
#             k = 0
#             for line in csvReader:
#                 k += 1
#                 image_name = line[0]
#                 label = line[5:]
#
#                 for i in range(14):
#                     if label[i]:
#                         a = float(label[i])
#                         if a == 1:
#                             label[i] = 1
#                         elif a == -1:
#                             if policy == "ones":
#                                 label[i] = 1
#                             elif policy == "zeroes":
#                                 label[i] = 0
#                             else:
#                                 label[i] = 0
#                         else:
#                             label[i] = 0
#                     else:
#                         label[i] = 0
#
#                 image_names.append('../' + image_name)
#                 labels.append(label)
#
#         self.image_names = image_names
#         self.labels = labels
#         self.transform = transform
#
#     def __getitem__(self, index):
#         """Take the index of item and returns the image and its labels"""
#
#         image_name = self.image_names[index]
#         image = Image.open(image_name).convert('RGB')
#         label = self.labels[index]
#         if self.transform is not None:
#             image = self.transform(image)
#         return image, torch.FloatTensor(label)
#
#     def __len__(self):
#         return len(self.image_names)