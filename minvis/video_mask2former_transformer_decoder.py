# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import fvcore.nn.weight_init as weight_init

from detectron2.config import configurable
from detectron2.layers import Conv2d

from mask2former.modeling.transformer_decoder.maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from mask2former.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine

from mask2former_video.modeling.transformer_decoder.video_mask2former_transformer_decoder import CrossAttentionLayer, FFNLayer, SelfAttentionLayer, VideoMultiScaleMaskedTransformerDecoder, MLP
import einops


@TRANSFORMER_DECODER_REGISTRY.register()
class VideoMultiScaleMaskedTransformerDecoder_frame(VideoMultiScaleMaskedTransformerDecoder):

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        # video related
        num_frames,
    ):
        super().__init__(
            in_channels=in_channels, 
            mask_classification=mask_classification,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            dec_layers=dec_layers,
            pre_norm=pre_norm,
            mask_dim=mask_dim,
            enforce_input_project=enforce_input_project,
            num_frames=num_frames,
        )

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        self.num_frames = num_frames

        # positional encoding
        # use 2D positional embedding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_block_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim*2,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_block_cross_attention_layers.append(
                 CrossAttentionLayer(
                     d_model=hidden_dim,
                     nhead=nheads,
                     dropout=0.0,
                     normalize_before=pre_norm,
                 )
             )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim*2,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim*2)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim*2)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim*2)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)


        self.recon_embed = nn.Sequential(
            MLP(hidden_dim, hidden_dim, hidden_dim * 4, 3),
            nn.Linear(hidden_dim*4, 28*28*3),
        )
        self.recon_conv = Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, x, mask_features, mask = None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []
        predictions_recon = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask, recon_query = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        predictions_recon.append(recon_query)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            block_mask = torch.zeros_like(attn_mask)
            random_idx = torch.randperm(attn_mask.shape[-1])[:attn_mask.shape[-1] // 2]
            block_mask[:, :, random_idx] = True

            spatial_output, appearance_output = output.split(output.shape[-1] // 2, dim=-1)
            spatial_embed, appearance_embed = query_embed.split(query_embed.shape[-1] // 2, dim=-1)

            # attention: cross-attention first
            spatial_output = self.transformer_cross_attention_layers[i](
                spatial_output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=spatial_embed
            )

            appearance_output = self.transformer_block_cross_attention_layers[i](
                 appearance_output, src[level_index],
                 memory_mask=block_mask,
                 memory_key_padding_mask=None,  # here we do not apply masking on padded region
                 pos=pos[level_index], query_pos=appearance_embed
            )
            
            output = torch.cat([spatial_output, appearance_output], dim=-1)

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask, recon_query = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_recon.append(recon_query)

        assert len(predictions_class) == self.num_layers + 1

        # expand BT to B, T  
        bt = predictions_mask[-1].shape[0]
        bs = bt // self.num_frames if self.training else 1
        t = bt // bs
        for i in range(len(predictions_mask)):
            predictions_mask[i] = einops.rearrange(predictions_mask[i], '(b t) q h w -> b q t h w', t=t)

        for i in range(len(predictions_recon)):
            predictions_recon[i] = einops.rearrange(predictions_recon[i], '(b t) q c h w -> b q t c h w', t=t)

        for i in range(len(predictions_class)):
            predictions_class[i] = einops.rearrange(predictions_class[i], '(b t) q c -> b t q c', t=t)

        pred_embds = self.decoder_norm(output)
        pred_embds = einops.rearrange(pred_embds, 'q (b t) c -> b c t q', t=t)

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'pred_recons': predictions_recon[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask, predictions_recon
            ),
            'pred_embds': pred_embds,
        }
        
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        spatial_output, appearance_output = decoder_output.split(decoder_output.shape[-1] // 2, dim=-1)
        outputs_class = self.class_embed(spatial_output)
        mask_embed = self.mask_embed(spatial_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        recon_query = self.recon_embed(appearance_output)
        recon_query = recon_query.reshape(-1, 28, 28, 3).permute(0, 3, 1, 2)
        recon_query = self.recon_conv(recon_query)
        recon_query = recon_query.reshape(*appearance_output.shape[:2], *recon_query.shape[1:])


        return outputs_class, outputs_mask, attn_mask, recon_query.sigmoid()
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_recon):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_recons": c}
                for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_recon[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
