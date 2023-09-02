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

        self.appearance_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

    def forward(self, output, appearance_features, output_mask=None):
        assert len(appearance_features) == self.num_feature_levels
        src = []
        pos = []
        attn_mask = []

        output_mask = output_mask.transpose(1,2).flatten(0,1) # B Q T H W -> BT Q H W
        
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

        output = self.appearance_embed(output)

        return output

