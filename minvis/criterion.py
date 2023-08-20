import logging
from itertools import chain

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from mask2former.utils.misc import is_dist_avail_and_initialized

from mask2former_video.modeling.criterion import VideoSetCriterion

class AppearanceSetCriterion(VideoSetCriterion):
    """
    This class computes the loss for MaskFormer with appearance features.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def loss_reid(self, outputs, targets, indices, num_masks):
        pred_embds = outputs["pred_embds"]
        tgt_embds = pred_embds[::2]
        cur_embds = pred_embds[1::2]

        dot_product = torch.einsum("bmc,bnc->bmn", [tgt_embds, cur_embds])
        dot_product = torch.exp(dot_product)
        pos_pair = torch.diagonal(dot_product, dim1=1, dim2=2)
        tgt_loss = -torch.log(pos_pair / torch.sum(dot_product, dim=2))
        cur_loss = -torch.log(pos_pair / torch.sum(dot_product, dim=1))

        contras_loss = torch.cat([tgt_loss, cur_loss]).mean()

        normalized_tgt_embds = F.normalize(tgt_embds, dim=2)
        normalized_cur_embds = F.normalize(cur_embds, dim=2)
        cos_sim = torch.einsum("bmc,bnc->bmn", [normalized_tgt_embds, normalized_cur_embds])

        cos_label = torch.eye(cos_sim.shape[1], device=cos_sim.device)[None, :, :]
        cos_loss = torch.abs((cos_sim - cos_label)**2).mean()

        loss = {
            "loss_reid": contras_loss,
            "loss_aux_cos": cos_loss,
        }

        return loss


    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'reid': self.loss_reid,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v[::2] for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets[::2])

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                video_aux_outputs = {k: v[::2] for k, v in aux_outputs.items()}
                indices = self.matcher(video_aux_outputs, targets[::2])
                for loss in self.losses:
                    if loss in ["reid"]: continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses