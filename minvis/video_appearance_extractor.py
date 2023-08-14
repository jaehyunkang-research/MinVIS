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
        self.apperance_query_feat = nn.Embedding(1, hidden_dim)
        # learnable query p.e.
        self.appearance_query_embed = nn.Embedding(1, hidden_dim)

        self.feature_proj = Conv2d(hidden_dim, hidden_dim, kernel_size=1, norm=get_norm("GN", hidden_dim))

        self.embed_head = nn.Linear(hidden_dim, hidden_dim)
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
        B = pred_masks.shape[0]

        pos_embeds = torch.chunk(self.pe_layer(features, None).flatten(2).permute(2, 0, 1), B, dim=1)
        features = torch.chunk(self.feature_proj(features).flatten(2).permute(2, 0, 1), B, dim=1) # BT x C x H x W -> HW x BT x C

        if self.training:
            assert self.num_frames > 2
            pos_neg_label = torch.zeros((self.num_frames-1)*3, device=pred_masks.device)
            pos_neg_label[:self.num_frames-1] = 1
            pos_label = pos_neg_label == 1
            neg_label = pos_neg_label == 0

            contras_loss = 0
            aux_cosine_loss = 0
            all_instances = 0

            for bs, target in enumerate(targets):
                video_masks = F.interpolate(target['masks'], size=pred_masks.shape[-2:]) > 0.1
                valid_video = video_masks[:, 0].sum((-1,-2)) != 0
                video_masks = video_masks[valid_video]

                num_instances = video_masks.shape[0]
                if num_instances == 0:
                    continue
                anchor_masks = video_masks[:, :1]
                pos_masks = video_masks[:, 1:]
                hard_neg_masks = anchor_masks.expand_as(pos_masks) & ~pos_masks
                soft_neg_masks = ~hard_neg_masks & ~pos_masks

                attn_mask = torch.cat([anchor_masks, pos_masks, hard_neg_masks, soft_neg_masks], dim=1).transpose(0, 1)
                attn_mask = ~(attn_mask.flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1))

                feature_batch = torch.tensor([1] + [3] * (self.num_frames-1), device=video_masks.device)
                feature = features[bs].repeat_interleave(feature_batch, dim=1)
                pos_embed = pos_embeds[bs].repeat_interleave(feature_batch, dim=1)

                query_embed = self.appearance_query_embed.weight.unsqueeze(1).repeat(num_instances, sum(feature_batch), 1)
                output = self.apperance_query_feat.weight.unsqueeze(1).repeat(num_instances, sum(feature_batch), 1)

                if len(torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])[0]):
                    print('hi')
                    # postive가 없거나, anchor가 없거나, negative가 없거나 등의 문제 해결 필요

                for i in range(self.num_layers):
                    output = self.transformer_cross_attention_layers[i](
                        output, feature,
                        memory_mask=attn_mask,
                        memory_key_padding_mask=None,
                        pos=pos_embed, query_pos=query_embed
                    )
                    
                    # FFN
                    output = self.transformer_ffn_layers[i](
                        output
                    )

                output = self.decoder_norm(output)

                anchor_embedding = self.embed_head(output[:, :1])
                pos_neg_embedding = self.embed_head(output[:, 1:])

                dot_product = torch.einsum('nkc,nrc->nkr', anchor_embedding, pos_neg_embedding).squeeze(1) # num of anchors (num instance) x num of pos/neg
                aux_normalize_pos_neg_embedding = F.normalize(pos_neg_embedding, dim=-1)
                aux_normalize_anchor_embedding = F.normalize(anchor_embedding, dim=-1)
                aux_cosine_similarity = torch.einsum('nkc,nrc->nkr', aux_normalize_anchor_embedding, aux_normalize_pos_neg_embedding).squeeze(1)

                pred_pos = dot_product * pos_label
                pred_neg = dot_product * neg_label
                
                pred_pos[:, neg_label] = pred_pos[:, neg_label] + float('inf')
                pred_neg[:, pos_label] = pred_neg[:, pos_label] + float('-inf')

                _pos_expand = torch.repeat_interleave(pred_pos, dot_product.shape[1], dim=1)
                _neg_expand = pred_neg.repeat(1, dot_product.shape[1])
                # [bz,N], N is all pos and negative samples on reference frame, label indicate it's pos or negative
                x = torch.nn.functional.pad(
                    (_neg_expand - _pos_expand), (0, 1), "constant", 0)
                try:
                    contras_loss += torch.logsumexp(x, dim=1).sum()
                except:
                    print('h')
                aux_cosine_loss += (torch.abs(aux_cosine_similarity - pos_neg_label) ** 2).mean()
                all_instances += num_instances

            loss = {
                'loss_reid': contras_loss / all_instances,
                'loss_aux_cosine': aux_cosine_loss / all_instances
                }
        else:
            self.num_frames = features.shape[1]
            attn_mask = pred_masks.transpose(1, 2).flatten(0, 1) # B x Q x T x H x W -> BT x Q x H x W

        return None, loss
    
    def forward_cl_loss(self, queries, targets):
        '''
        Args:
            queries: (B, T, Q, C)
            targets: (B, T, Q, C)
        
        Returns:
            loss: (B, T, Q)
        '''
        B, T, _, _ = queries.shape
        sample_idx = torch.block_diag(*torch.ones(T, self.num_queries, self.num_queries, device=queries.device))
        contrast = torch.matmul(queries.flatten(1,2), queries.flatten(1,2).transpose(1,2)) # B x TQ x TQ
        contrast = contrast[:, ~sample_idx.bool()].reshape(-1, T, self.num_queries, self.num_queries*(T-1)) # B x T x Q x (Q-1)
        num_instances = []
        contras_loss = 0
        for bs in range(B):
            num_instance = len(targets[bs*T]['labels'])
            if num_instance == 0:
                continue
            num_instances.append(num_instance*T)
            pos_idx = torch.eye(self.num_queries, device=queries.device)
            pos_idx = pos_idx.repeat(1, T-1).bool()
            pos_idx[num_instance:] = 0
            neg_idx = ~pos_idx
            neg_idx[num_instance:] = 0
            pos_embed = contrast[bs, :, pos_idx].reshape(T, num_instance, -1)
            neg_embed = contrast[bs, :, neg_idx].reshape(T, num_instance, -1)
            x = F.pad((neg_embed[:, :, None] - pos_embed[..., None]).flatten(2,), (0, 1), "constant", 0)
            contras_loss += torch.logsumexp(x, dim=2).sum()
        return contras_loss, sum(num_instances)