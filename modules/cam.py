import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, List, Optional, Tuple, Union
from torch import Tensor, nn
import numpy as np

class CAM:
    def __init__(self, normalized = False, relu = True):
        self.normalized = normalized
        self.relu = relu

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
        cams_max[cams_max<1e-12] = 1e-12
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


