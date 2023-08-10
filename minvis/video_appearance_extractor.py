import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, get_norm
from detectron2.utils.registry import Registry

from mask2former.modeling.transformer_decoder.maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from mask2former.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine

from mask2former_video.modeling.transformer_decoder.video_mask2former_transformer_decoder import CrossAttentionLayer, FFNLayer

APPERANCE_EXTRACTOR_REGISTRY = Registry("APPERANCE_EXTRACTOR")
APPERANCE_EXTRACTOR_REGISTRY.__doc__ = """
Registry for apperance extractor, which extract apperance features from input images.
"""

def build_appearance_extractor(cfg):
    """
    Build a apperance extractor from `cfg.MODEL.APPERANCE_EXTRACTOR.NAME`.
    """
    name = cfg.MODEL.APPERANCE_EXTRACTOR.NAME
    return APPERANCE_EXTRACTOR_REGISTRY.get(name)(cfg)

@APPERANCE_EXTRACTOR_REGISTRY.register()
class CrossAttentionExtractor(nn.Module):
    @configurable
    def __init__(
            self,
            hidden_dim: int,
            num_queries: int,
            nheads: int,
            dim_feedforward: int,
            dec_layers: int,
            pre_norm: bool,
            num_frames: int,
    ):
        super().__init__()
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        for _ in range(self.num_layers):
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.apperance_query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.appearance_query_embed = nn.Embedding(num_queries, hidden_dim)

        self.feature_proj = Conv2d(hidden_dim, hidden_dim, kernel_size=1, norm=get_norm("GN", hidden_dim))

        self.num_frames = num_frames

    @classmethod
    def from_config(cls, cfg):
        ret = {}

        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        # ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["dec_layers"] = 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["num_frames"] = cfg.INPUT.SAMPLING_FRAME_NUM

        return ret

    def forward(self, features: Tensor, pred_masks: Tensor):
        pos_embed = self.pe_layer(features, None).flatten(2).permute(2, 0, 1)
        features = self.feature_proj(features).flatten(2).permute(2, 0, 1) # BT x C x H x W -> HW x BT x C

        attn_mask = pred_masks.transpose(1, 2).flatten(0, 1) # B x Q x T x H x W -> BT x Q x H x W
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()

        _, bs, _ = features.shape

        query_embed = self.appearance_query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.apperance_query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        for i in range(self.num_layers):
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, features,
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos_embed, query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

        output = self.decoder_norm(output)

        return output.transpose(0, 1).reshape(-1, self.num_frames, self.num_queries, output.shape[-1])