import fvcore.nn.weight_init as weight_init
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, get_norm
from detectron2.utils.registry import Registry
from detectron2.utils.comm import get_world_size

from mask2former.modeling.transformer_decoder.maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from mask2former.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine
from mask2former.utils.misc import is_dist_avail_and_initialized

from mask2former_video.modeling.transformer_decoder.video_mask2former_transformer_decoder import MLP

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
        self.num_queries = num_queries
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        self.reid_convs = nn.ModuleList([
                Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, stride=2, norm=get_norm("GN", hidden_dim), activation=nn.ReLU(inplace=True)),
                Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, stride=2, norm=get_norm("GN", hidden_dim), activation=nn.ReLU(inplace=True)),
                Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, stride=2, norm=get_norm("GN", hidden_dim), activation=nn.ReLU(inplace=True)),
            ])

        for layer in self.reid_convs:
            weight_init.c2_xavier_fill(layer)

        self.reid_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 2)

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

    def forward(self, features: Tensor, pred_masks: Tensor, targets: Tensor = None):
        key_features, ref_features = features
        key_features = F.interpolate(self.feature_proj(key_features), scale_factor=0.5) # BT x C x H x W 

        roi_features = []
        feature_batch = []

        temperature = 1.0
        num_instances = 0

        loss = None

        if self.training:
            ref_features = F.interpolate(self.feature_proj(ref_features), scale_factor=0.5) 
            for bs, target in enumerate(targets):
                anchor_masks = F.interpolate(target["masks"], scale_factor=0.125)
                pos_masks = F.interpolate(target["flip_masks"], scale_factor=0.125)
                hard_neg_masks = F.interpolate((target["masks"].bool() & (~target["flip_masks"].bool())).float(), scale_factor=0.125)

                anchor_features = key_features[bs, None] * anchor_masks
                pos_features = ref_features[bs, None] * pos_masks
                hard_neg_features = ref_features[bs, None] * hard_neg_masks

                roi_features.append(torch.cat([anchor_features, pos_features, hard_neg_features], dim=0))
                feature_batch.append(roi_features[-1].shape[0])
            roi_features = torch.cat(roi_features, dim=0)

        else:
            pred_masks = F.interpolate(pred_masks[0], scale_factor=0.5)
            roi_features = pred_masks.unsqueeze(2) * key_features[None]
            roi_features = roi_features.flatten(0, 1)

        for conv in self.reid_convs:
            roi_features = conv(roi_features)
        roi_features = F.adaptive_avg_pool2d(roi_features, 1).flatten(1)
        roi_features = self.reid_embed(roi_features)

        if self.training:
            roi_features_per_image = torch.split(roi_features, feature_batch, dim=0)
            
            contras_loss = 0
            for roi_embds in roi_features_per_image:
                anchor_embds, pos_embds, hard_neg_embds = torch.chunk(roi_embds, 3, dim=0)
                if anchor_embds.shape[0] == 0:
                    continue
                num_instances += anchor_embds.shape[0]

                pos_neg_dot = torch.einsum("nc,mc->nm", [anchor_embds, pos_embds])
                hard_neg_dot = torch.einsum("nc,mc->nm", [anchor_embds, hard_neg_embds])
                
                pos_dot = pos_neg_dot.diag()
                soft_neg_dot = pos_neg_dot[~torch.eye(anchor_embds.shape[0]).bool()].reshape(anchor_embds.shape[0], -1)
                hard_neg_dot = hard_neg_dot.diag()[:, None]

                all_dot_product = torch.cat([soft_neg_dot, hard_neg_dot, pos_dot[:, None]], dim=1)
                all_dot_product = (all_dot_product - pos_dot[:, None]) / temperature
                contras_loss += torch.logsumexp(all_dot_product, dim=1).sum()

                

                loss = {'loss_reid': contras_loss / num_instances * 5}
        else:
            roi_features = roi_features.reshape(self.num_queries, -1, roi_features.shape[-1])

        return roi_features, loss
    
    def forward_cl_loss(self, queries, indices):
        '''
        Args:
            queries: (B, T, Q, C)
            targets: (B, T, Q, C)
        
        Returns:
            loss: (B, T, Q)
        '''
        B, T, Q, _ = queries.shape
        for i, indice in enumerate(indices):
            queries[i, 1] = queries[i, 1][indice]
        sample_idx = torch.block_diag(*torch.ones(T, Q, Q, device=queries.device))
        contrast = torch.matmul(queries.flatten(1,2), queries.flatten(1,2).transpose(1,2)) # B x TQ x TQ
        contrast = contrast[:, ~sample_idx.bool()].reshape(-1, T, Q, Q*(T-1)) # B x T x Q x (Q-1)
        contrast -= contrast.diagonal(dim1=2, dim2=3)[...,None]
        contras_loss = torch.logsumexp(contrast.flatten(0,1), dim=2).sum() / contrast.shape[:-1].numel()
        return contras_loss, 1