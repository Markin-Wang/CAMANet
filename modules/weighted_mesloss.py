from torch.nn import MSELoss
import torch

class Weighted_MSELoss(MSELoss):

    def forward(self, input, target, weights):
        assert input.shape == target.shape
        assert input.size(0) == weights.size(0)
        B, C = input.size(0), input.shape[-1]
        loss = torch.cdist(input, target, p=2).sum(dim=-1)
        loss = weights * loss
        loss = loss.sum() / (B*C)
        return loss