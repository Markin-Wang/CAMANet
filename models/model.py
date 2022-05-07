import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.my_encoder_decoder import EncoderDecoder as r2gen
from modules.standard_trans import EncoderDecoder as st_trans
from modules.cam_attn_con import  CamAttnCon
from modules.my_encoder_decoder import LayerNorm
from modules.forebacklearning import ForeBackLearning

class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer, logger = None, config = None):
        super(R2GenModel, self).__init__()
        self.args = args
        self.addcls = args.addcls
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args, logger, config)
        self.fbl = args.fbl
        self.wmse = args.wmse
        self.attn_cam = args.attn_cam
        if self.fbl:
            self.fore_back_learn = ForeBackLearning(norm=LayerNorm(self.visual_extractor.num_features) if args.norm_fbl else None)
        if self.attn_cam:
            self.attn_cam_con = CamAttnCon(method=args.attn_method)
        self.sub_back = args.sub_back
        self.records = []
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

    def forward(self, images, targets=None,labels=None, mode='train',vis = False):
        if vis:
            save_img = images[0].detach().cpu()
        fore_map, total_attns, weights = None, None, None
        if self.addcls:
            patch_feats, gbl_feats, logits, cams = self.visual_extractor(images)
            if self.fbl and labels is not None:
                fore_rep, back_rep, fore_map = self.fore_back_learn(patch_feats, cams, labels)
                if vis:
                    self.records.append([save_img,fore_map[0].detach().cpu(),labels[0].detach().cpu()])
                if self.sub_back:
                    patch_feats = patch_feats - back_rep
                patch_feats = torch.cat((fore_rep, patch_feats), dim=1)

        else:
            patch_feats, gbl_feats = self.visual_extractor(images)
        if mode == 'train':
            output, fore_rep_encoded, target_embed, align_attns = self.encoder_decoder(gbl_feats, patch_feats, targets, mode='forward')
            if self.addcls and self.attn_cam:
                fore_map, total_attns = self.attn_cam_con(fore_map, fore_rep_encoded, target_embed, align_attns)
                if self.wmse:
                    p_logits = torch.sigmoid(logits)
                    scores = (p_logits * labels).sum(dim=-1)
                    weights = scores / labels.sum(dim=-1)
                # print(weights)
        elif mode == 'sample':
            output, _ = self.encoder_decoder(gbl_feats, patch_feats, mode='sample')
        else:
            raise ValueError
        if self.addcls and mode == 'train':
            return output, logits, cams, fore_map, total_attns, weights
        return output





