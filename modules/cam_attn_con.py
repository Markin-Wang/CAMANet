from torch import nn
import torch
import torch.nn.functional as F

class CamAttnCon(nn.Module):
    # cam attention consistency
    def __init__(self, method = 'weighted_sum'):
        super(CamAttnCon, self).__init__()
        # weighted sum or max
        self.method = method
        self.sim = nn.CosineSimilarity(dim=2)

    def forward(self, fore_map, fore_rep_encoded, target_embed, align_attns):
        # how to process extra token
        attns = align_attns[-1][:,:,:,1:]
        attns = torch.mean(attns, dim=1)
        attns = F.softmax(attns, dim=2)
        fore_map = fore_map.squeeze(1)
        if self.method == 'weighted_sum':
            scores = torch.matmul(target_embed, fore_rep_encoded.unsqueeze(-1))
            weights = F.softmax(scores, dim=1).transpose(-1,-2)
            total_attn = torch.matmul(weights, attns).squeeze(1)
            fore_map = F.softmax(fore_map, dim=1)
        elif self.method == 'max':
            #scores = torch.matmul(target_embed, fore_rep_encoded.unsqueeze(-1))
            #weights = F.softmax(scores, dim=1)
            weights = self.sim(target_embed, fore_rep_encoded.unsqueeze(1)).unsqueeze(-1)
            attns = F.relu(weights * attns)
            total_attn, _ = torch.max(attns, dim = 1)
            # print(total_attn.shape, fore_map.shape)
            # print(total_attn[0], fore_map[0])
        else:
            raise NotImplementedError
        return fore_map, total_attn

