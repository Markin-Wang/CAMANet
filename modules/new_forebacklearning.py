from torch import nn
import torch
from copy import deepcopy as clone

class ForeBackLearning(nn.Module):
    def __init__(self, fore_t=0.5, back_t=0.3, norm=None,dropout=None):
        super(ForeBackLearning, self).__init__()
        self.norm = norm
        self.dropout = dropout
        if norm:
            self.fore_norm = norm
            self.back_norm = clone(norm)
        if dropout:
            self.fore_dropout = dropout
            self.back_dropout = clone(dropout)
        self.fore_t = fore_t
        self.back_t = back_t

    def forward(self,patch_feats,cam,logits):
        logits = torch.sigmoid(logits)
        labels = (logits >= 0.5).float()
        #print(labels[0])
        cam = labels.unsqueeze(-1) * cam
        dm, _ = torch.max(cam, dim=1, keepdim=True)
        fore_map, back_map = dm.clone(), dm.clone()
        # fore_idx = (fore_map <= self.fore_t).nonzero()
        # back_idx = (back_map >= self.back_t).nonzero()
        fore_map[fore_map <= self.fore_t] = 0.0
        back_map[back_map >= self.back_t] = 0.0
        #print((fore_map >= self.fore_t)[0])
        #back_map = 1-fore_map
        fore_rep = torch.matmul(fore_map, patch_feats)
        back_rep = torch.matmul(back_map, patch_feats)
        if self.norm:
            fore_rep = self.fore_norm(fore_rep)
            back_rep = self.back_norm(back_rep)
        if self.dropout:
            fore_rep = self.fore_dropout(fore_rep)
            back_rep = self.back_dropout(back_rep)
        return fore_rep, back_rep, dm