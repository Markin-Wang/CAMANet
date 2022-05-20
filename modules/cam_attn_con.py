from torch import nn
import torch
import torch.nn.functional as F

class CamAttnCon(nn.Module):
    # cam attention consistency
    def __init__(self, method = 'mean', topk = 0.1, layer_id = 2):
        super(CamAttnCon, self).__init__()
        # weighted sum or max
        self.method = method
        self.sim = nn.CosineSimilarity(dim=2)
        self.topk = topk
        self.layer_id = layer_id

    def forward(self, fore_map, fore_rep_encoded, target_embed, align_attns):
        # how to process extra token
        attns = align_attns[self.layer_id]
        attns = torch.mean(attns, dim=1)
        fore_map = fore_map.squeeze(1)
        weights = self.sim(target_embed, fore_rep_encoded.unsqueeze(1)).unsqueeze(-1)
        _, idxs = torch.topk(weights.squeeze(-1), k = int(self.topk*weights.shape[1]), dim = 1)
        attns = F.relu(weights * attns)
        attns = torch.gather(attns, dim = 1, index = idxs.unsqueeze(-1).expand(idxs.size(0),idxs.size(1),attns.shape[-1]))
        attns = self._normalize(attns)
        if self.method == 'mean':
            # scores = torch.matmul(target_embed, fore_rep_encoded.unsqueeze(-1))
            # weights = F.softmax(scores, dim=1).transpose(-1,-2)
            # total_attn = torch.matmul(weights, attns).squeeze(1)
            total_attn, _ = torch.mean(attns, dim=1)
            fore_map = F.softmax(fore_map, dim=1)
        elif self.method == 'max':
            #scores = torch.matmul(target_embed, fore_rep_encoded.unsqueeze(-1))
            #weights = F.softmax(scores, dim=1)
            total_attn, _ = torch.max(attns, dim = 1)
            # print(total_attn.shape, fore_map.shape)
            # print(total_attn[0], fore_map[0])
        else:
            raise NotImplementedError
        return fore_map, total_attn, idxs


    def _normalize(self, cams):
        """CAM normalization."""
        cams = cams - cams.min(-1, keepdim=True).values
        #cams.sub_(cams.min(-1).values[(..., None)])
        cams_max = cams.max(-1).values[(..., None)]
        cams_max = torch.clamp(cams_max, min = 1e-12, max = 1)
        cams = cams / cams_max
        return cams

