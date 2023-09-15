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
        num_classes: int,
    ):
        super().__init__()
        # only use res2 feature
        self.num_feature_levels = len(in_channels)
        self.input_proj = nn.ModuleList()
        for l in range(self.num_feature_levels):
            self.input_proj.append(nn.Sequential(
                Conv2d(in_channels[l], hidden_dim, kernel_size=1, norm=get_norm("GN", hidden_dim)),
                Conv2d(hidden_dim, hidden_dim, kernel_size=1, norm=get_norm("GN", hidden_dim)),
            ))
        self.mask_feature_proj = Conv2d(hidden_dim, hidden_dim, kernel_size=1, norm=get_norm("GN", hidden_dim))
        self.appearance_embd = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
        self.appearance_norm = nn.LayerNorm(hidden_dim)

        self.track_head = MLP(hidden_dim, hidden_dim, hidden_dim, 2)

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        # appearance decoder
        self.num_heads = nheads
        self.num_layers = appearance_layers
        self.appearance_self_attention_layers = nn.ModuleList()
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

            self.appearance_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

    def forward(self, pred_embds, appearance_features, output_masks, mask_features=None, indices=None, targets=None):
        B, T, Q, C = pred_embds.shape
        output_masks = output_masks.transpose(1, 2).flatten(0, 1).detach()
        output_masks = (output_masks.sigmoid() > 0.5).float()

        for i in range(self.num_feature_levels):
            proj_features = self.input_proj[i](appearance_features[i]).flatten(2)
            appearance_queries = torch.einsum('bqc,bcd->bqd', pred_embds.flatten(0, 1), proj_features).softmax(-1)
            appearance_queries = torch.einsum('bqd,bcd->bqc', appearance_queries, proj_features)
            appearance_queries = self.appearance_embd(appearance_queries)

        output = appearance_queries.transpose(0, 1)

        mask_features = self.mask_feature_proj(mask_features)

        for i in range(self.num_layers):
            # attention: cross-attention first
            output = self.appearance_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
            )
            output = self.appearance_ffn_layers[i](
                output
            )

        appearance_queries = output.reshape(Q, B, T, C).permute(1, 2, 0, 3)

        appearance_queries = self.appearance_norm(appearance_queries)

        if self.training:
            key_queries, ref_queries = appearance_queries.unbind(1)

            valid_indices = [t['ids'].squeeze(1) != -1 for t in targets]
            key_reid_embds = self.track_head(key_queries)
            ref_reid_embds = self.track_head(ref_queries)

            outputs_class = self.class_embed(torch.stack([key_queries, ref_queries], dim=1))
            contrastive_items = self.match(key_reid_embds, ref_reid_embds, indices, valid_indices)

            outputs_mask_embed = self.mask_embed(torch.stack([key_queries, ref_queries], dim=1)).flatten(0, 1)
            outputs_mask = torch.einsum('bqc,bchw->bqhw', outputs_mask_embed, mask_features)

            out = {
                'pred_logits': outputs_class.flatten(0, 1),
                'pred_masks': outputs_mask,
                'contrastive_items': contrastive_items,
            }

            return out
            
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

        return (dists, cos_dists, labels)
    