import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size

from mask2former.utils.misc import is_dist_avail_and_initialized

from mask2former_video.modeling.criterion import VideoSetCriterion


class AppearanceSetCriterion(VideoSetCriterion):
    """
    This class computes the loss for MaskFormer with appearance embedding.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss_reid(self, outputs, targets, indices, num_masks):
        appearance_embds = outputs['appearance_embds'].permute(0, 2, 3, 1) # B T Q C

        num_instances = 0
        contrass_loss = 0
        cos_loss = 0

        for i in range(0, len(indices), 2):
            key_indice = indices[i][0][indices[i][1].argsort()]
            ref_indice = indices[i+1][0][indices[i+1][1].argsort()]
            num_instance = len(key_indice) + len(ref_indice)
            if num_instance == 0: continue
            num_instances += num_instance
            
            key_embds = appearance_embds[i//2][0][key_indice]
            ref_embds = appearance_embds[i//2][1][ref_indice]

            dot_product = torch.einsum('kc, rc -> kr', key_embds, ref_embds)
            dot_product = torch.exp(dot_product)
            pos_pair = dot_product.diag()
            tgt_loss = -torch.log(pos_pair / torch.sum(dot_product, dim=1))
            cur_loss = -torch.log(pos_pair / torch.sum(dot_product, dim=0))

            contrass_loss += tgt_loss.sum() + cur_loss.sum()

            cosine_similarity = torch.einsum('kc, rc -> kr', F.normalize(key_embds, dim=1), F.normalize(ref_embds, dim=1))
            cos_label = torch.eye(cosine_similarity.shape[0]).cuda()

            cos_loss += 2 * (torch.abs(cos_label - cosine_similarity)**2).sum()

        if num_instances == 0:
            return {
                'loss_reid': appearance_embds.sum() * 0.,
                'loss_aux_cos': appearance_embds.sum() * 0.,
            }
        
        return {
            'loss_reid': contrass_loss / num_instances,
            'loss_aux_cos': cos_loss / num_instances,
        }


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
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

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
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss in ["reid"]: continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses