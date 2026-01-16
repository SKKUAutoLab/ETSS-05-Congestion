import math
import os
import sys
from typing import Iterable
import numpy as np
import cv2
import torch
import torchvision.transforms as standard_transforms
import util.misc as utils

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def visualization(samples, targets, pred, vis_dir, split_map=None):
    gts = [t['points'].tolist() for t in targets]
    pil_to_tensor = standard_transforms.ToTensor()
    restore_transform = standard_transforms.Compose([DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), standard_transforms.ToPILImage()])
    images = samples.tensors
    masks = samples.mask
    for idx in range(images.shape[0]):
        sample = restore_transform(images[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_vis = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()
        size = 2
        for t in gts[idx]:
            sample_vis = cv2.circle(sample_vis, (int(t[1]), int(t[0])), size, (0, 0, 255), -1) # red ground truth points
        for p in pred[idx]:
            sample_vis = cv2.circle(sample_vis, (int(p[1]), int(p[0])), size, (0, 255, 0), -1) # green prediction points
        if split_map is not None:
            imgH, imgW = sample_vis.shape[:2]
            split_map = (split_map * 255).astype(np.uint8)
            split_map = cv2.applyColorMap(split_map, cv2.COLORMAP_JET)
            split_map = cv2.resize(split_map, (imgW, imgH), interpolation=cv2.INTER_NEAREST)
            sample_vis = split_map * 0.9 + sample_vis
        if vis_dir is not None:
            valid_area = torch.where(~masks[idx])
            valid_h, valid_w = valid_area[0][-1], valid_area[1][-1]
            sample_vis = sample_vis[:valid_h+1, :valid_w+1]
            name = targets[idx]['image_path'].split('/')[-1].split('.')[0]
            cv2.imwrite(os.path.join(vis_dir, '{}_gt{}_pred{}.jpg'.format(name, len(gts[idx]), len(pred[idx]))), sample_vis)

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(torch.device('cuda')) # samples.tensors: [8, 3, 256, 256]
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
        gt_points = [target['points'] for target in targets] # [92, 2]
        outputs = model(samples, epoch=epoch, train=True, criterion=criterion, targets=targets)
        loss_dict, weight_dict, losses = outputs['loss_dict'], outputs['weight_dict'], outputs['losses']
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def cal_distance(sig_s,idn, target_set, outputs_set):
    tp = 0
    for i in range(len(idn[0])):
        img_h1 = sig_s[idn[1][i]][1]
        img_w1 = sig_s[idn[1][i]][0]
        img_h2 = sig_s[idn[1][i]][3]
        img_w2 = sig_s[idn[1][i]][2]
        img_h = img_h2-img_h1
        img_w = img_w2-img_w1
        dis = math.sqrt(img_h * img_h + img_w * img_w) / 2
        if (target_set[idn[1][i]][0] - outputs_set[idn[0][i]][0]) * (target_set[idn[1][i]][0] - outputs_set[idn[0][i]][0]) + \
           (target_set[idn[1][i]][1] - outputs_set[idn[0][i]][1]) * (target_set[idn[1][i]][1] - outputs_set[idn[0][i]][1]) <= dis * dis:
            tp += 1
    return tp

@torch.no_grad()
def evaluate(model, data_loader, criterion=None, vis_dir=None, args=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter=" ")
    header = 'Test:'
    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)
    print_freq = 10
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(torch.device('cuda')) # [1, 3, 768, 1024]
        img_h, img_w = samples.tensors.shape[-2:]
        outputs = model(samples, criterion= criterion, test=True, targets=targets)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0] # [420]
        outputs_points = outputs['pred_points'][0] # [420, 2]
        outputs_offsets = outputs['pred_offsets'][0] # [420, 2]
        predict_cnt = len(outputs_scores)
        gt_cnt = targets[0]['points'].shape[0]
        target_set = targets[0]['points'] # [172, 2]
        outputs_set = outputs['pred_points'][0] # [420, 2]
        outputs_set[:, 0] *= img_h
        outputs_set[:, 1] *= img_w
        if args.type_dataset == 'carpk':
            target_box = targets[0]['bboxs']
            ind = outputs['ind'][0]
            if outputs_set.shape[0] != 0:
              tp = cal_distance(target_box, ind, target_set, outputs_set)
            else:
                tp = 0
        else:
            tp = 0
        fn = gt_cnt - tp
        fp = predict_cnt-tp
        mae = abs(predict_cnt - gt_cnt)
        abs_ = abs(predict_cnt - gt_cnt) / gt_cnt
        mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
        Precision_s = tp / (tp + fp + 0.0000000000000001)
        Recall_s = tp / (tp + fn + 0.000000000001)
        F1_s = 2 * (Precision_s * Recall_s) / (Precision_s + Recall_s + 0.00000001)
        results = {}
        toTensor = lambda x: torch.tensor(x).float().cuda()
        results['mae'], results['mse'], results['Precision_s'], results['Recall_s'], results['F1_s'], results['abs'] = toTensor(mae), toTensor(mse), Precision_s, Recall_s, F1_s,abs_
        metric_logger.update(mae=results['mae'], mse=results['mse'], Precision_s=results['Precision_s'], Recall_s=results['Recall_s'], F1_s=results['F1_s'], abs=results['abs'])
        results_reduced = utils.reduce_dict(results)
        metric_logger.update(mae=results_reduced['mae'], mse=results_reduced['mse'], Precision_s=results['Precision_s'], Recall_s=results['Recall_s'], F1_s=results['F1_s'], abs=results['abs'])
        if vis_dir: 
            points = [[point[0], point[1]] for point in outputs_points]
            split_map = (outputs['split_map_raw'][0].detach().cpu().squeeze(0) > 0.5).float().numpy()
            visualization(samples, targets, [points], vis_dir, split_map=split_map)
    metric_logger.synchronize_between_processes()
    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    results['mse'] = np.sqrt(results['mse'])
    return results