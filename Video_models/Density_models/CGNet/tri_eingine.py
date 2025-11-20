from typing import Iterable
import numpy as np
import torch
from utils import get_total_grad_norm, reduce_dict
from models.tri_sim_ot_b import similarity_cost
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, metric_logger: object, scaler: torch.cuda.amp.GradScaler, epoch, args):
    model.train()
    metric_logger.meters.clear()
    header = 'Epoch: [{}]'.format(epoch)
    metric_logger.set_header(header)
    for inputs, labels in metric_logger.log_every(data_loader):
        optimizer.zero_grad()
        for key in inputs.keys():
            inputs[key] = inputs[key].to(args.gpu)
        z1, z2, y1, y2 = model(inputs) # [1, 27, 37632], [1, 27, 37632], [1, 12, 37632], [1, 10, 37632]
        if args.distributed:
            loss_dict = model.module.loss(z1, z2, y1, y2)
        else:
            loss_dict = model.loss(z1, z2, y1, y2)
        all_loss = loss_dict["all"]
        loss_dict_reduced = reduce_dict(loss_dict)
        all_loss_reduced = loss_dict_reduced["all"]
        loss_value = all_loss_reduced.item()
        scaler.scale(all_loss).backward()
        scaler.unscale_(optimizer)
        if args.Misc.clip_max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.Misc.clip_max_norm)
        else:
            grad_total_norm = get_total_grad_norm(model.parameters(), args.Misc.clip_max_norm)
        scaler.step(optimizer)
        scaler.update()
        for k in loss_dict_reduced.keys():
            metric_logger.update(**{k: loss_dict_reduced[k]})
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_similarity(model, data_loader, metric_logger, args):
    model.eval()
    metric_logger.meters.clear()
    header = "Test"
    metric_logger.set_header(header)
    cnt = 0
    for inputs,labels in metric_logger.log_every(data_loader):
        cnt += 1
        for key in inputs.keys():
            inputs[key] = inputs[key].to(args.gpu)
        z1, z2, y1, y2 = model(inputs) # [1, 14, 37632], [1, 14, 37632], [1, 3, 37632], [1, 2, 37632]
        z1, z2, y1, y2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1), F.normalize(y1, dim=-1), F.normalize(y2, dim=-1)
        match_matrix = torch.bmm(torch.cat((z1, y1), dim=1), torch.cat((z2, y2), dim=1).transpose(1,2)) # [1, 17, 16]
        all_match = linear_sum_assignment(1-match_matrix.cpu().detach().numpy()[0]) # [16]
        pos_sim = 1 - similarity_cost(z1, z2).detach().cpu().numpy()[0] # [14, 14]
        pos_match = np.argmax(pos_sim, axis=1)
        if cnt % 100==0:
            print('Count: {}, Pos match: {}, All match: {}'.format(cnt, pos_match, all_match))
        pos_match_acc = pos_match==np.arange(pos_match.shape[0])
        all_match_acc = all_match[1][:pos_match.shape[0]]==np.arange(pos_match.shape[0])
        all_match_acc = all_match_acc.sum() / all_match_acc.shape[0]
        pos_match_acc = pos_match_acc.sum() / pos_match_acc.shape[0]
        metric_logger.update(pos_match_acc=pos_match_acc)
        metric_logger.update(all_match_acc=all_match_acc)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}