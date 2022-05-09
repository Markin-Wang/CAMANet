from torch import nn
import torch

class ForeBackLearning(nn.Module):
    def __init__(self, norm=None,dropout=None):
        super(ForeBackLearning, self).__init__()
        self.norm = norm
        self.dropout = dropout

    def forward(self,patch_feats,cam,labels):
        cam = labels.unsqueeze(-1) * cam
        fore_map, _ = torch.max(cam, dim=1, keepdim=True)
        back_map = 1-fore_map
        fore_rep = torch.matmul(fore_map, patch_feats)
        back_rep = torch.matmul(back_map, patch_feats)
        if self.norm:
            fore_rep = self.norm(fore_rep)
        if self.dropout:
            fore_rep = self.dropout(fore_rep)
        return fore_rep, back_rep, fore_map