import logging
import math
from typing import Tuple
import einops

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.utils.comm import get_world_size

from mask2former.utils.misc import is_dist_avail_and_initialized
from mask2former_video.modeling.criterion import VideoSetCriterion

class AppearanceCriterion(VideoSetCriterion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss_reid(self, outputs):
        dists, cos_dists, labels = outputs["contrastive_items"]
        
        loss_reid = 0.0
        loss_aux_cos = 0.0
        num_instances = sum([len(dist) for dist in dists])

        for dist, cos_dist, label in zip(dists, cos_dists, labels):
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
            loss_reid += loss.sum()

            loss = torch.abs(cos_dist - label.float())**2
            loss_aux_cos += loss.mean(-1).sum()

        return {'loss_reid': loss_reid / num_instances, 'loss_aux_cos': loss_aux_cos / num_instances}

    def forward(self, outputs, targets, indices):
        """
        Appearance-aware query loss computation
        """
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            if loss == "reid": losses.update(self.loss_reid(outputs))
            else: losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        return losses