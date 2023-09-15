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
        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

        # only use res2 feature
        self.num_feature_levels = len(in_channels)
        self.input_proj = nn.ModuleList()
        for l in range(self.num_feature_levels):
            self.input_proj.append(nn.Sequential(
                Conv2d(in_channels[l], hidden_dim, kernel_size=1, norm=get_norm("GN", hidden_dim)),
            ))
        self.appearance_embd = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        self.reid_embd = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        # appearance decoder
        self.num_heads = nheads

    def forward(self, pred_embds, appearance_features, output_masks, indices=None, targets=None):
        assert len(appearance_features) == self.num_feature_levels

        B, T, Q, C = pred_embds.shape
        output_masks = output_masks.squeeze(2).detach()

        for i in range(self.num_feature_levels):
            resize_mask = F.interpolate(output_masks, size=appearance_features[i].shape[-2:], mode='bilinear', align_corners=False).flatten(2).softmax(-1)
            proj_features = self.input_proj[i](appearance_features[i]).flatten(2)
            appearance_queries = torch.einsum('bqd,bcd->bqc', resize_mask, proj_features)

        appearance_queries = self.appearance_embd(appearance_queries).reshape(B, T, Q, C)

        reid_queries = self.reid_embd(pred_embds)

        if self.training:
            sample_appearance_queries = appearance_queries.unbind(1)
            sample_reid_queries = reid_queries.unbind(1)
            valid_indices = [t['ids'].squeeze(1) != -1 for t in targets]

            losses = dict()

            for t in range(1, T):
                sample_valid_indices = [(v1 & v2).cpu() for v1, v2 in zip(valid_indices[t-1::T], valid_indices[t::T])]
                ref_indices = [src[tgt.argsort()] for src, tgt in indices[t-1::T]]
                key_indices = [src[tgt.argsort()] for src, tgt in indices[t::T]]

                ref_queries = sample_reid_queries[t-1]
                key_queries = sample_reid_queries[t]
                
                dists, cos_dists, labels = self.match(key_queries, ref_queries, ref_indices, key_indices, sample_valid_indices)
                if len(dists) == 0:
                    loss = {f'loss_reid_{t}': key_queries.sum()*0., f'loss_aux_reid_{t}': ref_queries.sum()*0.}
                else:
                    loss = self.loss(dists, cos_dists, labels, t)
                losses.update(loss)

                ref_queries = sample_appearance_queries[t-1]
                key_queries = sample_appearance_queries[t]
                
                dists, cos_dists, labels = self.match(key_queries, ref_queries, ref_indices, key_indices, sample_valid_indices)
                if len(dists) == 0:
                    loss = {f'loss_aux_cos_{t}': ref_queries.sum()*0.}
                else:
                    loss = self.loss(dists, cos_dists, labels, t, reid=False)
                losses.update(loss)
            return losses

        return reid_queries, appearance_queries
    
    def match(self, key, ref, ref_indices, key_indices, valid_indices):
        dists, cos_dists, labels = [], [], []
        for key_embed, ref_embed, key_idx, ref_idx, valid_idx in zip(key, ref, key_indices, ref_indices, valid_indices):
            anchor = key_embed[key_idx[valid_idx]] 
            dist = torch.einsum('ac, kc -> ak', anchor, ref_embed)
            cos_dist = torch.einsum('ac, kc -> ak', F.normalize(anchor, dim=-1), F.normalize(ref_embed, dim=-1))
            label = ref_idx[valid_idx].to(anchor.device)
            label_ = torch.zeros_like(dist).scatter_(1, label[:, None], 1)

            dists.append(dist)
            cos_dists.append(cos_dist)
            labels.append(label_)

        return dists, cos_dists, labels
    
    def loss(self, dists, cos_dists, labels, t, reid=True):
        
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

            if reid:
                _pos_expand = torch.repeat_interleave(dist_pos, dist.shape[1], dim=1)
                _neg_expand = dist_neg.repeat(1, dist.shape[1])

                x = F.pad((_neg_expand - _pos_expand), (0,1), value=0)
                loss = torch.logsumexp(x, dim=1)# * (dist.shape[0] > 0).float()
                loss_reid += loss.sum()

            loss = torch.abs(cos_dist - label.float())**2
            loss_aux_cos += loss.mean(-1).sum()

        if reid:
            losses = {f'loss_reid_{t}': loss_reid / num_instances * 2, f'loss_aux_reid_{t}': loss_aux_cos / num_instances * 3}
        else:
            losses = {f'loss_aux_cos_{t}': loss_aux_cos / num_instances * 3}

        return losses

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