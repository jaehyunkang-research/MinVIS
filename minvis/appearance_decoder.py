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

        # only use res2 feature
        self.num_feature_levels = len(in_channels)
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for l in range(self.num_feature_levels):
            if in_channels[l] != hidden_dim:
                self.input_proj.append(Conv2d(in_channels[l], hidden_dim, kernel_size=1))
                weight_init.c2_msra_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        self.appearance_norm = nn.LayerNorm(hidden_dim)
        
        self.appearance_projection = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.appearance_prediction = MLP(hidden_dim, hidden_dim, hidden_dim, 2)

        self.criterion = nn.CosineSimilarity(dim=-1)

    def forward(self, output, appearance_features, output_mask, indices=None):
        assert len(appearance_features) == self.num_feature_levels
        src = []
        pos = []
        attn_mask = []

        output_mask = output_mask.squeeze(2) 
        
        for i in range(self.num_feature_levels):
            pos.append(self.pe_layer(appearance_features[i], None).flatten(2))
            src.append(self.input_proj[i](appearance_features[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            resize_mask = F.interpolate(output_mask, size=appearance_features[i].shape[-2:])
            attn_resize_mask = (resize_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,1) < 0.5).bool()
            attn_resize_mask[torch.where(attn_resize_mask.sum(-1) == attn_resize_mask.shape[-1])] = False
            attn_mask.append(attn_resize_mask)

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

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

        output_z = self.appearance_projection(self.appearance_norm(output))
        output_p = self.appearance_prediction(output_z)

        if self.training:
            loss = self.loss(output_z, output_p, indices)
            return {"loss_appearance": loss * 5}
            
        return output_z
    
    def loss(self, z, p, indices):
        T = 2 # only use two frames
        B = len(indices) // T
        idx0, idx1 = self._get_permutation_idx(indices)

        z0, z1 = z.transpose(0,1).reshape(B, T, -1, z.shape[-1]).unbind(1)
        p0, p1 = p.transpose(0,1).reshape(B, T, -1, p.shape[-1]).unbind(1)
        z0, z1 = z0[idx0].detach(), z1[idx1].detach()
        p0, p1 = p0[idx0], p1[idx1]

        loss = -(self.criterion(p0, z1).mean() + self.criterion(p1, z0).mean()) * 0.5
        return loss

    def _get_permutation_idx(self, indices):
        # permute targets following indices
        sorted_idx = [src[tgt.argsort()] for src, tgt in indices]
        batch_idx0 = torch.cat([torch.full_like(src, i) for i, src in enumerate(sorted_idx[::2])])
        src_idx0 = torch.cat([src for src in sorted_idx[::2]])
        batch_idx1 = torch.cat([torch.full_like(src, i) for i, src in enumerate(sorted_idx[1::2])])
        src_idx1 = torch.cat([src for src in sorted_idx[1::2]])
        return (batch_idx0, src_idx0), (batch_idx1, src_idx1)