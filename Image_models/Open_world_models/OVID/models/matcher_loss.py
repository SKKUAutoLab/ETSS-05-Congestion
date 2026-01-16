import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.matcher import HungarianMatcher_Crowd
import torch.distributed as dist

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

class Focal_L1(nn.Module):
    def __init__(self):
        super(Focal_L1, self).__init__()
        self.epsilon = torch.tensor(0.5)

    def forward(self, x, y):
        if isinstance(x, (float, int)):
            x = Variable(torch.Tensor([x]))
        if isinstance(y, (float, int)):
            y = Variable(torch.Tensor([y]))
        mse_loss = torch.mul(torch.abs(x - y) / (y + self.epsilon), torch.log2(F.mse_loss(x, y, reduction='none') + 1))
        return mse_loss

class SetCriterion_Crowd(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_points):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_points(self, outputs, targets, indices, num_points):
        assert 'pred_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        target_points = torch.cat([t['point'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.mse_loss(src_points, target_points, reduction='none')
        losses = {}
        losses['loss_point'] = loss_bbox.sum() / num_points
        return losses

    def loss_nums(self, outputs, targets, indices, num_points):
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1]
        threshold = 0.5
        predict_cnt = sum(int((t > threshold).sum()) for t in outputs_scores)
        focal_l1 = Focal_L1()
        focal_l1.cuda()
        loss=focal_l1(predict_cnt,num_points)
        losses = {}
        losses['loss_nums'] = loss.cuda()
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {'labels': self.loss_labels, 'points': self.loss_points, 'nums': self.loss_nums}
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward_and_pair(self, outputs, targets):
        output1 = {'pred_logits': outputs['pred_logits'], 'pred_points': outputs['pred_points']}
        indices1 = self.matcher(output1, targets)
        temp = []
        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(output1.values())).device) # [1]
        threshold = 0.5
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1] # [8, 9216]
        mask=outputs_scores > threshold
        for i in range(outputs_scores.shape[0]):
            temp.append(outputs['pred_points'][i][mask[i]])
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_boxes = torch.clamp(num_points / get_world_size(), min=1).item()
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, output1, targets, indices1, num_boxes))
        return losses, temp

def criterion():
    num_classes = 1
    matcher = HungarianMatcher_Crowd()
    weight_dict = {'loss_ce': 1, 'loss_points': 0.05, 'loss_nums': 0.05}
    losses = ['labels', 'points','nums']
    criterion = SetCriterion_Crowd(num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=0.5, losses=losses)
    return criterion.cuda()