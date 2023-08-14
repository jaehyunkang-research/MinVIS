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
        features = self.feature_proj(features) # BT x C x H x W 

        BT, Q, _, _, _ = pred_masks.shape
        if not self.training:
            self.num_frames = BT

        pred_masks = F.interpolate(pred_masks.squeeze(2), scale_factor=0.25).sigmoid().detach()
        reid_fearuemap = F.interpolate(self.feature_proj(features), scale_factor=0.25)
        roi_features = (reid_fearuemap[:, None] * pred_masks[:, :, None]).flatten(0,1)

        for conv in self.reid_convs:
            roi_features = conv(roi_features)
        roi_features = F.adaptive_avg_pool2d(roi_features, 1).flatten(1)
        roi_features = roi_features.flatten(1)
        # for fc in self.reid_fcs:
        #     roi_features = fc(roi_features)
        output = self.reid_embed(roi_features).reshape(BT//self.num_frames, self.num_frames, Q, -1)
        loss = None
        if self.training:
            # output = torch.gather(output, 2, mask_indices.reshape(-1, self.num_frames, self.num_queries, 1).expand_as(output))
            loss_reid, num_instances = self.forward_cl_loss(output, targets)

            # num_instances = torch.as_tensor(
            #     [num_instances], dtype=torch.float, device=output.device
            # )
            # if is_dist_avail_and_initialized():
            #     torch.distributed.all_reduce(num_instances)
            # num_instances = torch.clamp(num_instances / get_world_size(), min=1).item()

            loss = {'loss_reid': loss_reid / num_instances}

        return output, loss
    
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