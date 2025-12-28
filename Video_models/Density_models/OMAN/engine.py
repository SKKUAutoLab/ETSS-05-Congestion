import math
import os
import sys
from typing import Iterable
import numpy as np
import torch
import torch.nn.functional as F
import util.misc as utils
from util.misc import MetricLogger
from datasets.Locator_dataset import trans_dataset
from util.misc import nested_tensor_from_tensor_list
from scipy.optimize import linear_sum_assignment
from test import read_pts as read_pts_sense
from test_ht21 import read_pts as read_pts_ht21

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, epoch: int, max_norm: float = 0):
    metric_logger = MetricLogger({"delimiter": "\t", "print_freq": 25, "header": ""})
    metric_logger.meters.clear()
    iteration = 0
    header = 'Epoch: [{}]'.format(epoch)
    metric_logger.set_header(header)
    for inputs, labels in metric_logger.log_every(data_loader):
        iteration += 1
        samples, targets = trans_dataset(inputs, 'train')
        samples = samples.to(torch.device('cuda')) # [2, 3, 256, 256]
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
        gt_points = [target['points'] for target in targets] # [4, 2]
        outputs = model(samples, inputs, labels, epoch=epoch, train=True, criterion=criterion, targets=targets)
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

@torch.no_grad()
def evaluate(args, model, data_loader, epoch=0):
    model.eval()
    metric_logger = MetricLogger({"delimiter": "\t", "print_freq": 1, "header": ""})
    metric_logger.meters.clear()
    header = 'Epoch: [{}]'.format(epoch)
    metric_logger.set_header(header)
    video_results = {}
    gt_video_num_list = []
    gt_video_len_list = []
    pred_video_num_list = []
    interval = 15
    for imgs, labels in metric_logger.log_every(data_loader):
        cnt_list = []
        video_name = labels["video_name"][0]
        img_names = labels["img_names"]
        if args.type_dataset == 'SENSE':
            pos0, feature0 = read_pts_sense(model, imgs[0, 0]) # [30, 2], [1, 256, 160, 256]
        elif args.type_dataset == 'HT21':
            pos0, feature0 = read_pts_ht21(model, imgs[0, 0])
        else:
            print('This dataset does not exist')
            raise NotImplementedError
        if args.distributed:
            z0 = model.module.forward_single_image(imgs[0, 0].cuda().unsqueeze(0), [pos0], feature0, True) # [30, 256, 3, 3]
        else:
            z0 = model.forward_single_image(imgs[0, 0].cuda().unsqueeze(0), [pos0], feature0, True)
        pre_z = z0
        pre_pos = pos0
        cnt_0 = len(pos0)
        cum_cnt = cnt_0
        cnt_list.append(cnt_0)
        selected_idx = [v for v in range(interval, len(img_names), interval)]
        pos_lists = []
        inflow_lists = []
        outflow_lists = []
        pos_lists.append(pos0)
        inflow_lists.append([1 for _ in range(len(pos0))])
        for i in selected_idx:
            if args.type_dataset == 'SENSE':
                pos, feature1 = read_pts_sense(model, imgs[0, i]) # [30, 2], [1, 256, 160, 256]
            elif args.type_dataset == 'HT21':
                pos, feature1 = read_pts_ht21(model, imgs[0, i])
            else:
                print('This dataset does not exist')
                raise NotImplementedError
            pre_pre_z = pre_z
            if args.distributed:
                z1, z2, pre_z = model.module.forward_single_image(imgs[0, i].cuda().unsqueeze(0), [pos], feature1, True, pre_z) # [30, 1, 2313], [38, 1, 2313], [38, 256, 3, 3]
            else:
                z1, z2, pre_z = model.forward_single_image(imgs[0, i].cuda().unsqueeze(0), [pos], feature1, True, pre_z)
            z1 = F.normalize(z1, dim=-1).transpose(0, 1) # [1, 30, 2313]
            z2 = F.normalize(z2, dim=-1).transpose(0, 1) # [1, 38, 2313]
            match_matrix = torch.bmm(z1, z2.transpose(1, 2)) # [1, 30, 38]
            C = match_matrix.cpu().detach().numpy()[0]
            row_ind, col_ind = linear_sum_assignment(-C)
            sim_feat = z1[:, row_ind, :] * z2[:, col_ind, :] # [1, 30, 2313]
            if args.distributed:
                pred_logits = model.module.vic.regression(sim_feat.squeeze(0)) # [30, 2]
            else:
                pred_logits = model.vic.regression(sim_feat.squeeze(0))
            pred_prob = F.softmax(pred_logits, dim=1) # [30, 2]
            pred_score, pred_class = pred_prob.max(dim=1)
            pedestrian_list = col_ind[(1 - pred_class).bool().cpu().numpy()]
            pre_pedestrian_list = row_ind[(1 - pred_class).bool().cpu().numpy()]
            inflow_idx_list = [i for i in range(len(pos)) if i not in pedestrian_list]
            outflow_idx_list = [i for i in range(len(pre_pos)) if i not in pre_pedestrian_list]
            if inflow_idx_list:
                for idx2 in inflow_idx_list:
                    for idx1 in range(z1.shape[1]):
                        sim_feat2 = z1[:, idx1, :] * z2[:, idx2, :]
                        if args.distributed:
                            pred_logits2 = model.module.vic.regression(sim_feat2.squeeze(0))
                        else:
                            pred_logits2 = model.vic.regression(sim_feat2.squeeze(0))
                        pred_prob2 = F.softmax(pred_logits2, dim=0)
                        pred_score2, pred_class2 = pred_prob2.max(dim=0)
                        if pred_class2 == 0:
                            pedestrian_list = np.append(pedestrian_list, idx2)
                            pre_pedestrian_list = np.append(pre_pedestrian_list, idx1)
                            break
            inflow_idx_list = [i for i in range(len(pos)) if i not in pedestrian_list]
            outflow_idx_list = [i for i in range(len(pre_pos)) if i not in pre_pedestrian_list]
            pos_lists.append(pos)
            inflow_list = []
            for j in range(len(pos)):
                if j in inflow_idx_list:
                    inflow_list.append(1)
                else:
                    inflow_list.append(0)
            inflow_lists.append(inflow_list)
            cum_cnt += len(inflow_idx_list)
            cnt_list.append(len(inflow_idx_list))
            outflow_list = []
            for j in range(len(pre_pos)):
                if j in outflow_idx_list:
                    outflow_list.append(1)
                else:
                    outflow_list.append(0)
            outflow_lists.append(outflow_list)
            z_mask = np.array(outflow_list, dtype=bool) # [30]
            mem = pre_pre_z[0][:len(pre_pos)][z_mask] # [7, 256, 3, 3]
            pre_z = [torch.cat((pre_z[0], mem), dim=0)] # [45, 256, 3, 3]
            pre_pos = pos
        pos_lists = [pos_lists[i].tolist() for i in range(len(pos_lists))]
        video_results[video_name] = {"video_num": cum_cnt, "first_frame_num": cnt_0, "cnt_list": cnt_list, "frame_num": len(img_names), "pos_lists": pos_lists, "inflow_lists": inflow_lists}
        if args.type_dataset == 'SENSE':
            anno_path = os.path.join(args.ann_dir, video_name + ".txt")
            with open(anno_path, "r") as f:
                lines = f.readlines()
                all_ids = set()
                for line in lines:
                    line = line.strip().split(" ")
                    data = [float(x) for x in line[3:] if x != ""]
                    if len(data) > 0:
                        data = np.array(data) # [238]
                        data = np.reshape(data, (-1, 7)) # [34, 7]
                        ids = data[:, 6].reshape(-1, 1) # [34, 1]
                        for id in ids:
                            all_ids.add(int(id[0]))
        elif args.type_dataset == 'HT21':
            anno_path = os.path.join(args.input_dir, 'val', video_name + "/gt/gt.txt")
            with open(anno_path, "r") as f:
                lines = f.readlines()
                all_ids = set()
                file_idx = 0
                for line in lines:
                    line = line.split(',')
                    data = line
                    data = np.array(data)
                    if file_idx != int(line[0]):
                        file_idx = int(line[0])
                        if len(data) > 0:
                            ids = data[1].astype(int)
                            all_ids.add(ids)
        else:
            print('This dataset does not exist')
            raise NotImplementedError
        info = video_results[video_name]
        gt_video_num = len(all_ids)
        pred_video_num = info["video_num"]
        pred_video_num_list.append(pred_video_num)
        gt_video_num_list.append(gt_video_num)
        gt_video_len_list.append(info["frame_num"])
        mae_vic = abs((pred_video_num - gt_video_num) / gt_video_num)
        predict_cnt = len(pos)
        gt_cnt = len(labels["pts"][0][0].numpy())
        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
        results = {}
        toTensor = lambda x: torch.tensor(x).float().cuda()
        results['mae'], results['mse'] = toTensor(mae), toTensor(mse)
        results['mae_vic'] = toTensor(mae_vic)
        metric_logger.update(mae=results['mae'], mse=results['mse'], mae_vic=results['mae_vic'])
        results_reduced = utils.reduce_dict(results)
        metric_logger.update(mae=results_reduced['mae'], mse=results_reduced['mse'], mae_vic=results['mae_vic'])
    metric_logger.synchronize_between_processes()
    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    results['mse'] = np.sqrt(results['mse'])
    return results
