import math

import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from timm.models.layers import trunc_normal_
import torch.nn.functional as F

from modules.visual_extractor import VisualExtractor

import torchvision.models as models
from swintrans.models import build_model
from vit.models.modeling import VisionTransformer, CONFIGS
from modules.utils import load_pretrained
import numpy as np

class Classifier(nn.Module):
    def __init__(self, args, logger = None, config = None, n_classes = 14):
        super(Classifier, self).__init__()
        self.ve_name = args.ve_name
        self.dataset_name = args.dataset_name
        if args.ve_name == 'swin_transformer':
            # config.defrost()
            # config.MODEL.SWIN.DEPTHS = config.MODEL.SWIN.DEPTHS[:-1]
            # config.freeze()
            self.model = build_model(config)
            load_pretrained(config, self.model, logger)
            # print(self.model.layers)
            # self.model.layers = self.model.layers[:-1]
            # print(self.model.layers)
            # self.model.num_layers = self.model.num_layers - 1
            # self.model.num_features = int(self.model.embed_dim * 2 ** (self.model.num_layers - 1))
            self.num_features = self.model.num_features
        elif args.ve_name.lower().startswith('resnet101'):
            model = getattr(models, args.ve_name)(pretrained=True)
            self.num_features = model.fc.in_features
            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)
        elif args.ve_name.lower().startswith('vit'):
            config = CONFIGS[args.ve_name]
            self.model = VisionTransformer(config, args.img_size, zero_head=True, num_classes = n_classes)
            self.model.load_from(np.load(args.pretrained))
            self.model.head = None
            self.num_features = config.hidden_size
        elif args.ve_name.lower().startswith('densenet'):
            model = getattr(models, args.ve_name)(pretrained=True)
            self.num_features = model.classifier.in_features
            self.model = model.features
            self.avg_fnt = torch.nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        if args.dataset_name=='iu_xray':
            self.head = Linear(self.num_features, n_classes)
        else:
            self.head = Linear(self.num_features, n_classes)
        trunc_normal_(self.head.weight, std= 1/math.sqrt(self.num_features * n_classes))
        nn.init.constant_(self.head.bias, 0)


    def forward(self, images, labels = None, mode ='train'):
        if self.dataset_name == 'iu_xray':
            if self.ve_name.lower().startswith('vit'):
                feats_1, attn_weights_1 = self.model.forward_patch_features(images[:, 0])
                feats_2, attn_weights_2 = self.model.forward_patch_features(images[:, 0])
                feats = torch.cat((feats_1, feats_2), dim=1)
                patch_feats, avg_feats = feats[:, 1:, :], feats[:, 0, :]
            elif self.ve_name == 'swin_transformer':
                patch_feats_1 = self.model.forward_patch_features(images[:, 0])
                patch_feats_2 = self.model.forward_patch_features(images[:, 1])
                patch_feats = torch.cat((patch_feats_1, patch_feats_2), dim=1)
                avg_feats = torch.mean(patch_feats, -2)
            elif self.ve_name.startswith('resnet'):
                patch_feats_1 = self.model(images[:, 0])
                patch_feats_2 = self.model(images[:, 1])
                avg_feats_1 = F.adaptive_avg_pool2d(patch_feats_1, (1, 1)).squeeze().reshape(-1, patch_feats_1.size(1))
                avg_feats_2 = F.adaptive_avg_pool2d(patch_feats_2, (1, 1)).squeeze().reshape(-1, patch_feats_2.size(1))
                batch_size, feat_size, _, _ = patch_feats_1.shape
                patch_feats_1 = patch_feats_1.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
                patch_feats_2 = patch_feats_2.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
                patch_feats = torch.cat((patch_feats_1, patch_feats_2), dim=1)
                avg_feats = torch.cat((avg_feats_1, avg_feats_2), dim=1)
            elif self.ve_name.startswith('densenet'):
                patch_feats_1 = F.relu(self.model(images[:, 0]), inplace=True)
                patch_feats_2 = F.relu(self.model(images[:, 1]), inplace=True)
                #print(1111, torch.cat((patch_feats_1, patch_feats_2),dim=3).shape)
                # avg_feats_1 = F.adaptive_avg_pool2d(patch_feats_1, (1, 1)).squeeze().reshape(-1, patch_feats_1.size(1))
                # avg_feats_2 = F.adaptive_avg_pool2d(patch_feats_2, (1, 1)).squeeze().reshape(-1, patch_feats_2.size(1))

                #avg_feats = (avg_feats_1 + avg_feats_2)/2
                avg_feats = F.adaptive_avg_pool2d(torch.cat((patch_feats_1, patch_feats_2), dim=3), (1, 1)).squeeze().reshape(-1, patch_feats_1.size(1))
                # batch_size, feat_size, _, _ = patch_feats_1.shape
                # patch_feats_1 = patch_feats_1.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
                # patch_feats_2 = patch_feats_2.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
                # patch_feats = torch.cat((patch_feats_1, patch_feats_2), dim=1)
                # avg_feats = torch.cat((avg_feats_1, avg_feats_2), dim=1)

        else:
            if self.ve_name.lower().startswith('vit'):
                feats, attn_weights = self.model.forward_patch_features(images)
                patch_feats, avg_feats = feats[:, 1:, :], feats[:, 0, :]
            elif self.ve_name == 'swin_transformer':
                patch_feats = self.model.forward_patch_features(images)
                avg_feats = torch.mean(patch_feats, -2)
            elif self.ve_name.startswith('resnet'):
                patch_feats = self.model(images)
                avg_feats = F.adaptive_avg_pool2d(patch_feats, (1, 1)).squeeze().reshape(-1, patch_feats.size(1))
                batch_size, feat_size, _, _ = patch_feats.shape
                patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
            elif self.ve_name.startswith('densenet'):
                patch_feats = F.relu(self.model(images), inplace=True)
                avg_feats = F.adaptive_avg_pool2d(patch_feats, (1, 1)).squeeze().reshape(-1, patch_feats.size(1))
                batch_size, feat_size, _, _ = patch_feats.shape
                patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        logits = self.head(avg_feats)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits
