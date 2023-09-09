import logging
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, get_norm

from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP
from mask2former.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine
import einops

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

        # only use res2 feature
        self.num_feature_levels = len(in_channels)
        self.input_proj = nn.ModuleList()
        for l in range(self.num_feature_levels):
            self.input_proj.append(nn.Sequential(
                Conv2d(in_channels[l], hidden_dim, kernel_size=1, norm=get_norm("GN", hidden_dim)),
                Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, norm=get_norm("GN", hidden_dim)),
            ))
        self.appearance_norm = nn.LayerNorm(hidden_dim)
        self.appearance_aggregation = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3)
        self.appearance_embd = MLP(hidden_dim, hidden_dim, hidden_dim, 2)

        self.track_head = MLP(hidden_dim, hidden_dim, hidden_dim, 2)

        self.gating_head = MLP(hidden_dim, hidden_dim, 1, 1)

        self.criterion = nn.CosineSimilarity(dim=-1)

        # appearance decoder
        self.num_heads = nheads
        self.num_layers = appearance_layers

    def forward(self, pred_embds, appearance_features, output_masks, indices=None, targets=None):
        assert len(appearance_features) == self.num_feature_levels

        B, T, Q, C = pred_embds.shape
        output_masks = output_masks.transpose(1, 2).flatten(0, 1).detach()

        output = pred_embds.flatten(0, 1)


        appearance_queries = []
        for i in range(self.num_layers):
            appearance_feature = self.input_proj[i](appearance_features[i]).flatten(2)
            appearance_query = torch.einsum('bqc, bcd -> bqd', output, appearance_feature).softmax(-1)
            appearance_query = torch.einsum('bqd, bcd -> bqc', appearance_query, appearance_feature)
            appearance_queries.append(appearance_query.flatten(0,1))

        appearance_queries = self.appearance_aggregation(torch.stack(appearance_queries, dim=2)).reshape(B, T, Q, C)

        query_gating = self.gating_head(appearance_queries).sigmoid()

        appearance_queries = self.appearance_embd(self.appearance_norm(appearance_queries))

        if self.training:
            key_queries, ref_queries = appearance_queries.unbind(1)
            key_pred_embds, ref_pred_embds = pred_embds.unbind(1)
            key_gating, ref_gating = query_gating.unbind(1)

            valid_indices = [t['ids'].squeeze(1) != -1 for t in targets]
            key_queries = key_queries * (1 - key_gating) + key_pred_embds * key_gating
            ref_queries = ref_queries * (1 - ref_gating) + ref_pred_embds * ref_gating
            key_queries = self.track_head(key_queries)
            ref_queries = self.track_head(ref_queries)

            dists, cos_dists, labels = self.match(key_queries, ref_queries, indices, valid_indices)
            if len(dists) == 0:
                loss = {'loss_reid': key_queries.sum()*0., 'loss_aux_cos': ref_queries.sum()*0.}
            else:
                loss = self.loss(dists, cos_dists, labels)
            return loss
            
        appearance_queries = appearance_queries * (1 - query_gating) + pred_embds * query_gating # 여기 반대임!!!!!!
        track_queries = self.track_head(appearance_queries)

        return track_queries
    
    def match(self, key, ref, indices, valid_indices):
        sorted_idx = [src[tgt.argsort()] for src, tgt in indices]
        dists, cos_dists, labels = [], [], []
        for i, (key_embed, ref_embed) in enumerate(zip(key, ref)):
            b = 2 * i
            valid_idx = (valid_indices[b] & valid_indices[b+1]).cpu()
            anchor = key_embed[sorted_idx[b][valid_idx]] 
            dist = torch.einsum('ac, kc -> ak', anchor, ref_embed)
            cos_dist = torch.einsum('ac, kc -> ak', F.normalize(anchor, dim=-1), F.normalize(ref_embed, dim=-1))
            label = sorted_idx[b+1][valid_idx].to(anchor.device)
            label_ = torch.zeros_like(dist).scatter_(1, label[:, None], 1)

            dists.append(dist)
            cos_dists.append(cos_dist)
            labels.append(label_)

        return dists, cos_dists, labels
    
    def loss(self, dists, cos_dists, labels):
        
        loss_reid = 0.0
        loss_aux_cos = 0.0
        num_instances = sum([len(dist) for dist in dists])

        for dist, cos_dist, label in zip(dists, cos_dists, labels):
            pos_inds = (label == 1)
            neg_inds = (label == 0)
            dist_pos = dist * pos_inds.float()
            dist_neg = dist * neg_inds.float()
            dist_pos[neg_inds] = dist_pos[neg_inds] + float('inf')
            dist_neg[pos_inds] = dist_neg[pos_inds] - float('inf')

            _pos_expand = torch.repeat_interleave(dist_pos, dist.shape[1], dim=1)
            _neg_expand = dist_neg.repeat(1, dist.shape[1])

            x = F.pad((_neg_expand - _pos_expand), (0,1), value=0)
            loss = torch.logsumexp(x, dim=1)# * (dist.shape[0] > 0).float()
            loss_reid += loss.sum()

            loss = torch.abs(cos_dist - label.float())**2
            loss_aux_cos += loss.mean(-1).sum()

        return {'loss_reid': loss_reid / num_instances * 2, 'loss_aux_cos': loss_aux_cos / num_instances * 3}

    def _get_permutation_idx(self, indices, valid_indices):
        # permute targets following indices
        sorted_idx = [src[tgt.argsort()] for src, tgt in indices]
        split_idx = []
        labels = []
        for i in range(0, len(sorted_idx), 2):
            valid_idx = valid_indices[i].cpu()
            sorted_idx[i] = sorted_idx[i][valid_idx]
            sorted_idx[i+1] = sorted_idx[i+1][valid_idx]
            label = valid_indices[i+1][valid_idx]

            split_idx.append(valid_idx.sum())
            if len(label) == 0: continue
            labels.append(label)
        batch_idx0 = torch.cat([torch.full_like(src, i) for i, src in enumerate(sorted_idx[::2])])
        src_idx0 = torch.cat([src for src in sorted_idx[::2]])
        batch_idx1 = torch.cat([torch.full_like(src, i) for i, src in enumerate(sorted_idx[1::2])])
        src_idx1 = torch.cat([src for src in sorted_idx[1::2]])
        return (batch_idx0, src_idx0), (batch_idx1, src_idx1), split_idx, labels