import os
import numpy as np
import torch
import cv2
import torchvision.transforms as standard_transforms
from PIL import Image
import torchvision.utils as vutils

def calculate_median(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n % 2 == 1:
        median = sorted_data[n // 2]
    else:
        median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    return median

def softmax(x):
    x = np.array(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def cla_weight(gt_counts):
    median = calculate_median(gt_counts)
    ds = [ (value + 1.02) /  (median + 1.02) for value in gt_counts]
    weights = [ 1 /  np.log(d + 1.02) for d in ds]
    weights = softmax(weights)
    return weights

def eval_mc(pred_map, gt_map, log_para):
    pred_map = pred_map.squeeze().cpu().numpy()
    gt_map = gt_map.squeeze().cpu().numpy()
    class_num, H, W = gt_map.shape
    assert pred_map.shape == gt_map.shape
    gt_counts = []
    abs_errors = list()
    square_errors = list()
    for i in range(class_num):
        pred_cnt = pred_map[i,:,:].sum() / log_para
        gt_count = gt_map[i,:,:].sum() / log_para
        gt_counts.append(gt_count)
        abs_error = abs(gt_count-pred_cnt)
        square_error = (gt_count-pred_cnt)*(gt_count-pred_cnt)
        abs_errors.append(abs_error)
        square_errors.append(square_error)
    weights = cla_weight(gt_counts)
    return abs_errors, square_errors, weights

def adjust_learning_rate(optimizer,  cur_iters, max_iters, warmup='linear',  warmup_iters=500, warmup_ratio=1e-6, lr=0.00006, power=1.0, min_lr=0.0):
    if warmup is not None:
        if warmup not in ['constant', 'linear', 'exp']:
            raise ValueError(f'"{warmup}" is not a supported type for warming up, valid types are "constant" and "linear"')
    if warmup is not None:
        assert warmup_iters > 0, '"warmup_iters" must be a positive integer'
        assert 0 < warmup_ratio <= 1.0, '"warmup_ratio" must be in range (0,1]'
    if warmup == 'linear':
        k = (1 - cur_iters / warmup_iters) * (1 - warmup_ratio)
        warmup_lr = lr * (1 - k)
    if warmup == 'constant':
        warmup_lr = lr * warmup_ratio
    if warmup == 'exp':
        k = warmup_ratio**(1 - cur_iters / warmup_iters)
        warmup_lr = lr * k
    if warmup is None or  cur_iters >= warmup_iters:
        coff = (1 - cur_iters/max_iters)**power
        lr = (lr - min_lr) * coff + min_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr
        return warmup_lr

def adjust_double_learning_rate(optimizer, cur_iters, max_iters, warmup='linear', warmup_iters=1500, warmup_ratio=1e-6, base_lr=0.00006, conv_lr=0.000006, power=1.0, min_lr=0.0):
    if warmup is not None:
        if warmup not in ['constant', 'linear', 'exp']:
            raise ValueError(f'"{warmup}" is not a supported type for warming up, valid types are "constant" and "linear"')
    if warmup is not None:
        assert warmup_iters > 0, '"warmup_iters" must be a positive integer'
        assert 0 < warmup_ratio <= 1.0, '"warmup_ratio" must be in range (0,1]'
    if warmup == 'linear':
        k = (1-cur_iters/warmup_iters)*(1-warmup_ratio)
        warmup_base_lr = base_lr*(1-k)
        warmup_conv_lr = conv_lr*(1-k)
    if warmup == 'constant':
        warmup_base_lr = base_lr*warmup_ratio
        warmup_conv_lr = conv_lr*warmup_ratio
    if warmup == 'exp':
        k = warmup_ratio**(1-cur_iters/warmup_iters)
        warmup_base_lr = base_lr*k
        warmup_conv_lr = conv_lr*k
    if warmup is None or  cur_iters >= warmup_iters:
        coff = (1-cur_iters/max_iters)**power
        lr1 = (base_lr - min_lr)*coff + min_lr
        lr2 = (conv_lr - min_lr)*coff + min_lr
        for i_p, param in enumerate(optimizer.param_groups,0):
            if i_p<2:
                optimizer.param_groups[i_p]['lr'] = lr2
            elif i_p<5:
                optimizer.param_groups[i_p]['lr'] = lr1
            else:
                print('Invalid lr schedule setting!')
        return lr1, lr2
    else:
        for i_p, param in enumerate(optimizer.param_groups,0):
            if i_p<2:
                optimizer.param_groups[i_p]['lr'] = warmup_conv_lr
            elif i_p<5:
                optimizer.param_groups[i_p]['lr'] = warmup_base_lr
            else:
                print('Invalid lr schedule setting!')
        return warmup_base_lr, warmup_conv_lr

def is_validation(stages,freqs,cur_epoch):
    flag = False
    stage = []
    for i_stage, cur_stage in enumerate(stages,0):
        if cur_epoch >= cur_stage:
            stage = i_stage
    if cur_epoch % freqs[stage] == 0:
        flag = True
    return flag

def vis_results(vis_dir, epoch, restore, img, pred_map, gt_map, factor):
    pil_to_tensor = standard_transforms.ToTensor()
    x = []
    for idx, tensor in enumerate(zip(img, pred_map, gt_map)):
        if idx > 1:
            break
        pil_input = restore(tensor[0])
        w, h = pil_input.size
        pil_input = pil_input.resize((w // factor, h // factor), Image.BILINEAR)
        pred_color_map = cv2.applyColorMap((255 * tensor[1] / (tensor[2].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        gt_color_map = cv2.applyColorMap((255 * tensor[2] / (tensor[2].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        pil_label = Image.fromarray(cv2.cvtColor(gt_color_map, cv2.COLOR_BGR2RGB))
        pil_output = Image.fromarray(cv2.cvtColor(pred_color_map, cv2.COLOR_BGR2RGB))
        x.extend([pil_to_tensor(pil_input.convert('RGB')), pil_to_tensor(pil_label.convert('RGB')), pil_to_tensor(pil_output.convert('RGB'))])
    x = torch.stack(x, 0)
    x = vutils.make_grid(x, nrow=3, padding=5)
    x = (x.numpy() * 255).astype(np.uint8)
    x = torch.from_numpy(x).permute(1, 2, 0).cpu().detach().numpy()
    cv2.imwrite(os.path.join(vis_dir, str(epoch) + '_' + str(epoch + 1) + '.png'), x)

def update_model(net, epoch, exp_path, scores, train_record):
    val_loss_avg, overall_mae, overall_mse, cls_avg_mae, cls_avg_mse, cls_weight_mse = scores
    snapshot_name = 'all_ep_%d_cls_avg_mae_%.1f_cls_avg_mse_%.1f_cls_weight_mse_%.1f' % (epoch + 1, cls_avg_mae, cls_avg_mse, cls_weight_mse)
    if cls_avg_mae < train_record['best_cls_avg_mae'] or cls_avg_mse < train_record['best_cls_avg_mse'] or cls_weight_mse < train_record['best_cls_weight_mse']:   
        train_record['best_model_name'] = snapshot_name
        to_saved_weight = net.state_dict()
        torch.save(to_saved_weight, os.path.join(exp_path, snapshot_name + '.pth'))
    if cls_avg_mae < train_record['best_cls_avg_mae']:           
        train_record['best_cls_avg_mae'] = cls_avg_mae
        train_record['overall_mae'] = overall_mae
    if cls_avg_mse < train_record['best_cls_avg_mse']:           
        train_record['best_cls_avg_mse'] = cls_avg_mse
        train_record['overall_mae'] = overall_mse
    if cls_weight_mse < train_record['best_cls_weight_mse']:           
        train_record['best_cls_weight_mse'] = cls_weight_mse
        train_record['overall_mae'] = overall_mse
    return train_record

def save_checkpoint(trainer):
    latest_state = {'train_record':trainer.train_record, 'net':trainer.net.state_dict(), 'optimizer':trainer.optimizer.state_dict(), 'epoch': trainer.epoch,
                    'i_tb':trainer.i_tb, 'num_iters':trainer.num_iters, 'exp_path':trainer.exp_path}
    if not os.path.exists(trainer.exp_path):
        os.makedirs(trainer.exp_path)
    torch.save(latest_state,os.path.join(trainer.exp_path, 'latest_state.pth'))

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.cur_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val
        self.count += 1
        self.avg = self.sum / self.count

class AverageCategoryMeter(object):
    def __init__(self,num_class):
        self.num_class = num_class
        self.reset()

    def reset(self):
        self.cur_val = np.zeros(self.num_class)
        self.avg = np.zeros(self.num_class)
        self.sum = np.zeros(self.num_class)
        self.count = np.zeros(self.num_class)

    def update(self, cur_val, class_id):
        self.cur_val[class_id] = cur_val
        self.sum[class_id] += cur_val
        self.count[class_id] += 1
        self.avg[class_id] = self.sum[class_id] / self.count[class_id]

def train_collate(batch):
    transposed_batch = list(zip(*batch))
    rgb_list = transposed_batch[0]
    ir_list = transposed_batch[1]
    gt_list = transposed_batch[2]
    # mask_list = transposed_batch[3]
    rgb = torch.stack(rgb_list, 0)
    ir = torch.stack(ir_list, 0)
    gt_map = torch.stack(gt_list, 0)
    # mask_map = torch.stack(mask_list, 0)
    data = dict()
    data['rgb'] = rgb
    data['nir'] = ir
    data['gt_map'] = gt_map
    # data['mask_map'] = mask_map
    return data