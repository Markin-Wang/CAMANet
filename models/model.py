import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.my_encoder_decoder import EncoderDecoder as r2gen
from modules.standard_trans import EncoderDecoder as st_trans
from modules.cam_attn_con import  CamAttnCon
from modules.my_encoder_decoder import LayerNorm
from modules.old_forebacklearning import ForeBackLearning

class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer, logger = None, config = None):
        super(R2GenModel, self).__init__()
        self.args = args
        self.addcls = args.addcls
        self.vis = args.vis
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args, logger, config)
        self.fbl = args.fbl
        self.wmse = args.wmse
        self.attn_cam = args.attn_cam
        if self.fbl:
            #self.fore_back_learn = ForeBackLearning(fore_t=args.fore_t, back_t=args.back_t, norm=LayerNorm(self.visual_extractor.num_features))
            self.fore_back_learn = ForeBackLearning(norm=LayerNorm(self.visual_extractor.num_features))
        if self.attn_cam:
            self.attn_cam_con = CamAttnCon(method=args.attn_method, topk= args.topk, layer_id=args.layer_id, vis=args.vis)
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

    def forward(self, images, targets=None,labels=None, mode='train'):
        fore_map, total_attns, weights, attns, idxs, align_attns_train = None, None, None, None, None, None
        if self.addcls:
            patch_feats, gbl_feats, logits, cams = self.visual_extractor(images)
            #if self.fbl and labels is not None:
            if self.fbl:
                fore_rep, back_rep, fore_map = self.fore_back_learn(patch_feats, cams, logits)
                if self.sub_back:
                    patch_feats = patch_feats - back_rep
                patch_feats = torch.cat((fore_rep, patch_feats), dim=1)

        else:
            patch_feats, gbl_feats = self.visual_extractor(images)
        if mode == 'train':
            output, fore_rep_encoded, target_embed, align_attns = self.encoder_decoder(gbl_feats, patch_feats, targets, mode='forward')
            if self.addcls and self.attn_cam:
                total_attns, idxs, align_attns_train = self.attn_cam_con(fore_rep_encoded, target_embed, align_attns, targets)
                # print(weights)
        elif mode == 'sample':
            output, _, attns = self.encoder_decoder(gbl_feats, patch_feats, mode='sample')
        else:
            raise ValueError
        if mode == 'train':
            if self.addcls:
                return output, logits, cams, fore_map, total_attns, idxs, align_attns_train
            else:
                return output
        return output, attns



