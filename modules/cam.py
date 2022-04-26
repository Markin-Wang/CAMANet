import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, List, Optional, Tuple, Union
from torch import Tensor, nn

class CAM:
    def __init__(self, normalized = True, relu = False):
        self.normalized = normalized
        self.relu = relu

    def compute_scores(self,patch_feats, fc_layer, class_idx):
        weights = self._get_weights(fc_layer, class_idx)
        cams = []

        with torch.no_grad():
            for weight, activation in zip(weights, self.hook_a):
                missing_dims = activation.ndim - weight.ndim  # type: ignore[union-attr]
                weight = weight[(...,) + (None,) * missing_dims]

                # Perform the weighted combination to get the CAM
                cam = torch.nansum(weight * activation, dim=1)  # type: ignore[union-attr]

                if self.relu:
                    cam = F.relu(cam, inplace=True)

                # Normalize the CAM
                if self.normalized:
                    cam = self._normalize(cam)

                cams.append(cam)
        return cams

    @staticmethod
    @torch.no_grad()
    def _normalize(cams: Tensor, spatial_dims: Optional[int] = None) -> Tensor:
        """CAM normalization."""
        spatial_dims = cams.ndim - 1 if spatial_dims is None else spatial_dims
        cams.sub_(cams.flatten(start_dim=-spatial_dims).min(-1).values[(...,) + (None,) * spatial_dims])
        cams.div_(cams.flatten(start_dim=-spatial_dims).max(-1).values[(...,) + (None,) * spatial_dims])


    @torch.no_grad()
    def _get_weights(self,fc_layer, class_idx):
        fc_weights = fc_layer.weight.data
        if fc_weights.ndim > 2:
            fc_weights = fc_weights.view(*fc_weights.shape[:2])
        if isinstance(class_idx, int):
            return [fc_weights[class_idx, :].unsqueeze(0)]
        else:
            return [fc_weights[class_idx, :]]


