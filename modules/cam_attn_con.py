from torch import nn
import torch
import torch.nn.functional as F
import math

class CamAttnCon(nn.Module):
    # cam attention consistency
    def __init__(self, method = 'mean', topk = 0.1, layer_id = 2, vis = False):
        super(CamAttnCon, self).__init__()
        # weighted sum or max
        self.method = method
        self.sim = nn.CosineSimilarity(dim=2)
        self.topk = topk
        self.layer_id = layer_id
        self.vis = vis

    def forward(self, fore_rep_encoded, target_embed, align_attns, targets):
        # how to process extra token
        targets = targets.clone()[:, :-1]
        seq_mask = (targets.data > 0)
        seq_mask[:, 0] += True
        attns = align_attns[self.layer_id]
        attns = torch.mean(attns, dim=1)
        weights = self.sim(target_embed, fore_rep_encoded.unsqueeze(1))
        weights = weights.masked_fill(seq_mask == 0, -1)
        _, idxs = torch.topk(weights, k = int(self.topk*weights.shape[1]), dim = 1)
        weights = weights.unsqueeze(-1)
        attns = F.relu(weights * attns)
        seq_len = torch.sum(seq_mask, dim=1)
        true_topk = seq_len * self.topk
        if self.method == 'mean':
            # scores = torch.matmul(target_embed, fore_rep_encoded.unsqueeze(-1))
            # weights = F.softmax(scores, dim=1).transpose(-1,-2)
            # total_attn = torch.matmul(weights, attns).squeeze(1)
            total_attn = [self._normalize(torch.mean(attn[idxs[i][:math.ceil(true_topk[i])]], dim=0)).unsqueeze(0) for i, attn in enumerate(attns)]
            #total_attn, _ = torch.mean(attns, dim=1)
        elif self.method == 'max':
            #scores = torch.matmul(target_embed, fore_rep_encoded.unsqueeze(-1))
            #weights = F.softmax(scores, dim=1)
            # total_attn = [self._normalize(torch.max(attn[idxs[i][:math.ceil(true_topk[i])]], dim=0).values).unsqueeze(0) for i, attn in
            #          enumerate(attns)]
            total_attn = [self._normalize(torch.max(attn[idxs[i][:math.ceil(true_topk[i])]], dim=0).values).unsqueeze(0) for i, attn in
                     enumerate(attns)]
            total_attn = torch.cat(total_attn, dim=0)
            #total_attn, _ = torch.max(attns, dim = 1)
            # print(total_attn.shape, fore_map.shape)
            # print(total_attn[0], fore_map[0])
        else:
            raise NotImplementedError
        if self.vis:
            return total_attn, [idxs[i][:math.ceil(true_topk[i])].detach().cpu() for i in range(len(attns))], align_attns
        return total_attn, None, None


    def _normalize(self, cams):
        """CAM normalization."""
        cams = cams - cams.min(-1, keepdim=True).values
        #cams.sub_(cams.min(-1).values[(..., None)])
        cams_max = cams.max(-1).values[(..., None)]
        cams_max[cams_max<1e-12] = 1e-12
        cams = cams / cams_max
        return cams

