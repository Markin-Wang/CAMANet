import torch
import torch.nn as nn
import torchvision.models as models
from swintrans.models import build_model
from modules.utils import load_pretrained

class VisualExtractor(nn.Module):
    def __init__(self, args, logger = None, config = None):
        super(VisualExtractor, self).__init__()
        self.ve_name = args.ve_name
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
        elif args.ve_name == 'resnet101':
            model = getattr(models, args.ve_name)(pretrained=True)
            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)
            self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
            self.num_features = 2048

    def forward(self, images):
        if self.ve_name == 'swin_transformer':
            patch_feats = self.model.forward_patch_features(images)
            avg_feats = torch.mean(patch_feats, -2)
        elif self.ve_name == 'resnet101':
            patch_feats = self.model(images)
            avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
            batch_size, feat_size, _, _ = patch_feats.shape
            patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        # batch_size, feat_size, _, _ = patch_feats.shape
        # patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats
