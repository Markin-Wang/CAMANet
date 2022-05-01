import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder as r2gen
from modules.standard_trans import EncoderDecoder as st_trans

class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer, logger = None, config = None):
        super(R2GenModel, self).__init__()
        self.args = args
        self.addcls = args.addcls
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args, logger, config)
        self.fbl = args.fbl
        self.sub_back = args.sub_back
        if args.ed_name == 'r2gen':
            self.encoder_decoder = r2gen(args, tokenizer)
        elif args.ed_name == 'st_trans':
            self.encoder_decoder = st_trans(args, tokenizer)
        else:
            raise NotImplementedError
        # if args.dataset_name == 'iu_xray':
        #     self.forward = self.forward_iu_xray
        # else:
        #     self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    # def forward_iu_xray(self, images, targets=None, mode='train'):
    #     bs = images.shape[0]
    #     images_cat = torch.cat((images[:, 0], images[:, 1]), dim=0)
    #     #att_feats, fc_feats = self.visual_extractor(images_cat)
    #     att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
    #     att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
    #     fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
    #     att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
    #     #print('1111',att_feats.shape, fc_feats.shape)
    #     if mode == 'train':
    #         output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
    #     elif mode == 'sample':
    #         output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
    #     else:
    #         raise ValueError
    #     return output

    # def forward_mimic_cxr(self, images, targets=None, mode='train'):
    #     att_feats, fc_feats = self.visual_extractor(images)
    #     if mode == 'train':
    #         output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
    #     elif mode == 'sample':
    #         output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
    #     else:
    #         raise ValueError
    #     return output

    def forward(self, images, targets=None,labels=None, mode='train'):
        if self.addcls:
            patch_feats, gbl_feats, logits, cams = self.visual_extractor(images)
            if self.fbl:
                fore_rep, back_rep = self.ForeBackLearning(patch_feats, cams, labels)
                if self.sub_back:
                    patch_feats = patch_feats - back_rep
                patch_feats = torch.cat((fore_rep, patch_feats), dim=1)

        else:
            patch_feats, gbl_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(gbl_feats, patch_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(gbl_feats, patch_feats, mode='sample')
        else:
            raise ValueError
        if self.addcls:
            return output, logits, cams
        return output

    def ForeBackLearning(self,patch_feats,cam,labels):
        cam = labels.unsqueeze(-1) * cam
        fore_map, _ = torch.max(cam, dim=1, keepdim=True)
        back_map = 1-fore_map
        fore_rep = torch.matmul(fore_map, patch_feats)
        back_rep = torch.matmul(back_map, patch_feats)
        return fore_rep,back_rep


