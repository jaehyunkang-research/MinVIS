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
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        for l in range(self.num_feature_levels):
            self.input_proj.append(nn.Sequential(
                Conv2d(in_channels[l], hidden_dim, kernel_size=1, norm=get_norm("GN", hidden_dim)),
                Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, norm=get_norm("GN", hidden_dim)),
            ))
        self.appearance_norm = nn.LayerNorm(hidden_dim)
        self.appearance_embd = MLP(hidden_dim, hidden_dim, hidden_dim, 2)

        self.track_head = MLP(hidden_dim, hidden_dim, hidden_dim, 2)

        self.criterion = nn.CosineSimilarity(dim=-1)

        # appearance decoder
        self.num_heads = nheads
        self.num_layers = appearance_layers
        self.appearance_self_attention_layers = nn.ModuleList()
        self.appearance_cross_attention_layers = nn.ModuleList()
        self.appearance_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.appearance_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.appearance_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.appearance_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

    def forward(self, pred_embds, appearance_features, output_masks, indices=None):
        assert len(appearance_features) == self.num_feature_levels
        src, pos = [], []
        attn_mask = []

        B, C, T, Q = pred_embds.shape
        output_masks = output_masks.transpose(1, 2).flatten(0, 1)

        for i in range(self.num_feature_levels):
            pos.append(self.pe_layer(appearance_features[i], None).flatten(2))
            src.append(self.input_proj[i](appearance_features[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            resize_mask = F.interpolate(output_masks, size=appearance_features[i].shape[-2:])
            attn_resize_mask = (resize_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,1) < 0.5).bool()
            attn_resize_mask[torch.where(attn_resize_mask[:, :, None].sum(-1) == attn_resize_mask.shape[-1])] = False
            attn_mask.append(attn_resize_mask)

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        output = pred_embds.permute(3, 0, 2, 1).reshape(Q, B*T, C)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # attention: cross-attention first
            output = self.appearance_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask[level_index],
                memory_key_padding_mask=None,
                pos=pos[level_index],
            )
            output = self.appearance_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
            )
            output = self.appearance_ffn_layers[i](
                output
            )

        appearance_queries = output.reshape(Q, B, T, C).permute(1, 2, 0, 3)

        appearance_queries = self.appearance_embd(self.appearance_norm(appearance_queries))

        if self.training:
            key_queries, ref_queries = appearance_queries.unbind(1)

            key_idx, ref_idx = self._get_permutation_idx(indices)
            split_idx = [len(i[0]) for i in indices[::T]]

            key_queries, ref_queries = key_queries[key_idx], ref_queries[ref_idx]
            key_queries = self.track_head(key_queries)
            ref_queries = self.track_head(ref_queries)

            dists, cos_dists = self.match(key_queries, ref_queries, split_idx)
            if len(dists) == 0:
                loss = {'loss_reid': key_queries.sum()*0., 'loss_aux_cos': ref_queries.sum()*0.}
            else:
                loss = self.loss(dists, cos_dists)
            return loss
            
        track_queries = self.track_head(appearance_queries)

        return track_queries
    
    def match(self, key, ref, indices):
        key_queries = torch.split(key, indices)
        ref_queries = torch.split(ref, indices)

        dists, cos_dists = [], []
        for key_query, ref_query in zip(key_queries, ref_queries):
            if len(key_query) == 0: continue
            dist = torch.mm(key_query, ref_query.T)
            cos_dist = torch.mm(F.normalize(key_query, dim=-1), F.normalize(ref_query, dim=-1).T)
            dists.append(dist)
            cos_dists.append(cos_dist)

        return dists, cos_dists
    
    def loss(self, dists, cos_dists):
        
        loss_reid = 0.0
        loss_aux_cos = 0.0

        for dist, cos_dist in zip(dists, cos_dists):
            label = torch.eye(dist.shape[0], device=dist.device)

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
            loss_reid += loss.sum() / dist.shape[0]

            loss = torch.abs(cos_dist - label.float())**2
            loss_aux_cos += loss.sum() / dist.shape[0]

        return {'loss_reid': loss_reid / len(dists), 'loss_aux_cos': loss_aux_cos / len(dists)}

    def _get_permutation_idx(self, indices):
        # permute targets following indices
        sorted_idx = [src[tgt.argsort()] for src, tgt in indices]
        batch_idx0 = torch.cat([torch.full_like(src, i) for i, src in enumerate(sorted_idx[::2])])
        src_idx0 = torch.cat([src for src in sorted_idx[::2]])
        batch_idx1 = torch.cat([torch.full_like(src, i) for i, src in enumerate(sorted_idx[1::2])])
        src_idx1 = torch.cat([src for src in sorted_idx[1::2]])
        return (batch_idx0, src_idx0), (batch_idx1, src_idx1)