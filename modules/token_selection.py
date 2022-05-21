from torch import nn
import torch
from copy import deepcopy as clone
from typing import Any, List, Optional, Tuple, Union
from torch import Tensor, nn

class TokenSelection(nn.Module):
    def __init__(self, fore_t=0.5, back_t=0.3):
        super(TokenSelection, self).__init__()
        self.fore_t = fore_t
        self.back_t = back_t

    def forward(self,patch_feats,head,cls_idx,logits):
        cam = self.compute_scores(patch_feats, head, cls_idx)
        logits = torch.sigmoid(logits)
        labels = (logits >= 0.5).float()
        #print(labels[0])
        cam = labels.unsqueeze(-1) * cam
        dm, _ = torch.max(cam, dim=1, keepdim=True)
        # fore_map, back_map = dm.clone(), dm.clone()
        # fore_idx = (fore_map <= self.fore_t).nonzero()
        # back_idx = (back_map >= self.back_t).nonzero()
        # fore_map[fore_map <= self.fore_t] = 0.0
        # back_map[back_map >= self.back_t] = 0.0
        fore_idxs = dm > self.fore_t
        #print((fore_map >= self.fore_t)[0])
        #back_map = 1-fore_map
        # fore_rep = torch.matmul(fore_map, patch_feats)
        # back_rep = torch.matmul(back_map, patch_feats)
        # if self.norm:
        #     fore_rep = self.fore_norm(fore_rep)
        #     back_rep = self.back_norm(back_rep)
        # if self.dropout:
        #     fore_rep = self.fore_dropout(fore_rep)
        #     back_rep = self.back_dropout(back_rep)
        return fore_idxs

    def compute_scores(self,patch_feats, fc_layer, class_idx):
        weights = self._get_weights(fc_layer, class_idx)
        with torch.no_grad():
        # n_cam = weights.shape[0]
        #patch_feats = patch_feats.unsqueeze(1).expand(patch_feats.shape[0], n_cam, patch_feats.shape[1],patch_feats.shape[2])
            cams = torch.matmul(patch_feats, weights.transpose(-2,-1)).transpose(-2,-1)
        # print(cams.shape)
        #
        #
        #     for weight, activation in zip(weights, patch_feats):
        #         # missing_dims = activation.ndim - weight.ndim  # type: ignore[union-attr]
        #         # weight = weight[(...,) + (None,) * missing_dims]
        #
        #         # Perform the weighted combination to get the CAM
        #         cam = torch.nansum(weight * activation, dim=1)  # type: ignore[union-attr]


            if self.relu:
                cams = F.relu(cams, inplace=True)
        # Normalize the CAM
            if self.normalized:
                cams = self._normalize(cams)

            #cams.append(cam)
        return cams

    @staticmethod
    @torch.no_grad()
    def _normalize(cams: Tensor, spatial_dims: Optional[int] = None) -> Tensor:
        """CAM normalization."""
        cams.sub_(cams.min(-1).values[(..., None)])
        cams_max = cams.max(-1).values[(..., None)]
        cams_max = torch.clamp(cams_max, min = 1e-12, max = 1)
        cams.div_(cams_max)
        return cams


    @torch.no_grad()
    def _get_weights(self,fc_layer, class_idx):
        fc_weights = fc_layer.weight.data
        if fc_weights.ndim > 2:
            fc_weights = fc_weights.view(*fc_weights.shape[:2])
        if isinstance(class_idx, int):
            return fc_weights[class_idx, :].unsqueeze(0)
        else:
            return fc_weights[class_idx, :]