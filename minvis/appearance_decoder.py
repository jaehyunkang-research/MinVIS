import logging
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP
from mask2former.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine
import einops

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    indices = torch.full_like(tensor, torch.distributed.get_rank())
    indices_gather = [torch.zeros(1, dtype=torch.long, device=tensor.device)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(indices_gather, torch.as_tensor(tensor.shape[0], device=tensor.device), async_op=False)
    indices = torch.cat(indices_gather, dim=0)

    tensors_gather = [torch.ones(indices[i], tensor.shape[-1], device=tensor.device)
        for i in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)

    return output, indices
class AppearanceDecoder(nn.Module):

    def __init__(
        self, 
        in_channels,
        *,
        hidden_dim: int,
        nheads: int,
        dim_feedforward: int,
        appearance_layers: int,
        pre_norm: bool,
    ):
        super().__init__()

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # appearance decoder
        self.num_heads = nheads
        self.num_layers = appearance_layers

        # only use res2 feature
        self.num_feature_levels = len(in_channels)
        self.input_proj = nn.ModuleList()
        for l in range(self.num_feature_levels):
            self.input_proj.append(Conv2d(in_channels[l], hidden_dim, kernel_size=1))
            weight_init.c2_msra_fill(self.input_proj[-1])

        self.appearance_norm = nn.LayerNorm(hidden_dim)

        self.reid_projection = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.reid_prediction = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
        
        self.appearance_aggregation = MLP(hidden_dim*self.num_feature_levels, hidden_dim, hidden_dim, 2)
        self.appearance_projection = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.appearance_prediction = MLP(hidden_dim, hidden_dim, hidden_dim, 2)

        self.total_projection = MLP(hidden_dim*2, hidden_dim, hidden_dim, 3)
        self.total_prediction = MLP(hidden_dim, hidden_dim, hidden_dim, 2)

        self.criterion = nn.CosineSimilarity(dim=-1)

    def forward(self, output, appearance_features, output_mask, indices=None, targets=None):
        assert len(appearance_features) == self.num_feature_levels

        output_mask = output_mask.squeeze(2).detach()
        
        appearance_queries = []
        for i in range(self.num_feature_levels):
            appearance_feature = self.input_proj[i](appearance_features[i]).flatten(2)
            appearance_query = torch.einsum('qbc,bcd->qbd', output, appearance_feature).softmax(dim=-1)
            appearance_query = torch.einsum('qbd,bcd->qbc', appearance_query, appearance_feature)
            appearance_queries.append(appearance_query)

        appearance_queries = self.appearance_aggregation(torch.cat(appearance_queries, dim=-1))
        output_appearance_z = self.appearance_projection(self.appearance_norm(appearance_queries))
        output_appearance_p = self.appearance_prediction(output_appearance_z)

        output_reid_z = self.reid_projection(output)
        output_reid_p = self.reid_prediction(output_reid_z)

        output_z = self.total_projection(torch.cat([appearance_queries, output], dim=-1))
        output_p = self.total_prediction(output_z)

        if self.training:
            total_loss = self.loss(output_z, output_p, indices, targets)
            appearance_loss = self.loss(output_appearance_z, output_appearance_p, indices, targets)
            reid_loss = self.loss(output_reid_z, output_reid_p, indices)
            return {
                "loss_total": total_loss * 3, 
                "loss_appearance": appearance_loss, 
                "loss_reid": reid_loss}
            
        return output_z
    
    def loss(self, z, p, indices, targets=None):
        T = 2 # only use two frames
        B = len(indices) // T
        idx0, idx1 = self._get_permutation_idx(indices, targets)

        z0, z1 = z.transpose(0,1).reshape(B, T, -1, z.shape[-1]).unbind(1)
        p0, p1 = p.transpose(0,1).reshape(B, T, -1, p.shape[-1]).unbind(1)
        z0, z1 = z0[idx0].detach(), z1[idx1].detach()
        p0, p1 = p0[idx0], p1[idx1]

        with torch.cuda.amp.autocast(True):
            loss = self.contrastive_loss(p0, z1) + self.contrastive_loss(p1, z0)
        return loss
    
    def contrastive_loss(self, q, k):
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        k, idx = concat_all_gather(k)
        idx = torch.cumsum(idx, 0)

        logits = torch.einsum('nc,mc->nm', [q, k]) / 0.07
        N = logits.size(0)
        labels = torch.arange(idx[torch.distributed.get_rank()] - N, idx[torch.distributed.get_rank()], dtype=torch.long).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * 0.07)

    def _get_permutation_idx(self, indices, targets=None):
        # permute targets following indices
        sorted_idx = [src[tgt.argsort()] for src, tgt in indices]
        if targets:
            targets = [t['ids'] for t in targets]
            for i in range(0, len(targets), 2):
                valid = ((targets[i] != -1) & (targets[i+1] != -1)).squeeze(1).cpu()
                sorted_idx[i] = sorted_idx[i][valid]
                sorted_idx[i+1] = sorted_idx[i+1][valid]
        batch_idx0 = torch.cat([torch.full_like(src, i) for i, src in enumerate(sorted_idx[::2])])
        src_idx0 = torch.cat([src for src in sorted_idx[::2]])
        batch_idx1 = torch.cat([torch.full_like(src, i) for i, src in enumerate(sorted_idx[1::2])])
        src_idx1 = torch.cat([src for src in sorted_idx[1::2]])
        return (batch_idx0, src_idx0), (batch_idx1, src_idx1)