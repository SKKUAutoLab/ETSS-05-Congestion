import os
import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from model.locator import Crowd_locator
from config import cfg
from misc.utils import adjust_learning_rate, vis_results, AverageMeter, AverageCategoryMeter, update_model, print_NWPU_summary
import datasets
import cv2
from tqdm import tqdm
from misc.compute_metric import eval_metrics

class Trainer():
    def __init__(self, cfg_data, args):
        self.cfg_data = cfg_data
        self.args = args
        # train and val loader
        self.train_loader, self.val_loader, self.restore_transform = datasets.loading_data(args.type_dataset)
        # model
        self.net = Crowd_locator(cfg.NET, cfg.GPU_ID)
        # optimizer
        if cfg.OPT == 'Adam':
            self.optimizer = optim.Adam([{'params': self.net.Extractor.parameters(), 'lr': cfg.LR_BASE_NET, 'weight_decay':1e-5},
                                         {'params': self.net.Binar.parameters(), 'lr': cfg.LR_BM_NET}])
        else:
            print('This optimizer does not exist')
            raise NotImplementedError
        self.scheduler = StepLR(self.optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)
        self.train_record = {'best_F1': 0, 'best_Pre': 0,'best_Rec': 0, 'best_mae': 1e20, 'best_mse':1e20, 'best_nae':1e20, 'best_model_name': ''}
        self.epoch = 0
        self.i_tb = 0
        self.num_iters = cfg.MAX_EPOCH * np.int(len(self.train_loader))
        if cfg.RESUME:
            latest_state = torch.load(cfg.RESUME_PATH)
            self.net.load_state_dict(latest_state['net'])
            self.optimizer.load_state_dict(latest_state['optimizer'])
            self.scheduler.load_state_dict(latest_state['scheduler'])
            self.epoch = latest_state['epoch'] + 1
            self.i_tb = latest_state['i_tb']
            self.num_iters = latest_state['num_iters']
            self.train_record = latest_state['train_record']
            print('Load ckpt from:', cfg.RESUME_PATH)

    def forward(self):
        for epoch in range(self.epoch,cfg.MAX_EPOCH):
            self.epoch = epoch
            self.train()
            # if epoch % cfg.VAL_FREQ == 0 and epoch > cfg.VAL_DENSE_START:
            if epoch % 1 == 0:
                self.validate()

    def train(self):
        self.net.train()
        for i, data in enumerate(self.train_loader, 0):
            img, gt_map = data
            img = Variable(img).cuda() # [6, 3, 512, 1024]
            gt_map = Variable(gt_map).cuda() # [6, 1, 512, 1024]
            self.optimizer.zero_grad()
            threshold_matrix, pre_map, binar_map = self.net(img, gt_map) # [6, 1, 512, 1024], [6, 1, 512, 1024], [6, 1, 512, 1024]
            head_map_loss, binar_map_loss = self.net.loss
            all_loss =  head_map_loss + binar_map_loss
            all_loss.backward()
            self.optimizer.step()
            lr1, lr2 = adjust_learning_rate(self.optimizer, cfg.LR_BASE_NET, cfg.LR_BM_NET, self.num_iters, self.i_tb)
            if (i + 1) % cfg.PRINT_FREQ == 0:
                print('Epoch: {}, Iter: [{}/{}], Loss: {:.4f}, Lr1: {:.2f}, Lr2: {:.2f}'.format(self.epoch + 1, i + 1, len(self.train_loader), head_map_loss.item(),
                      self.optimizer.param_groups[0]['lr'] * 10000, self.optimizer.param_groups[1]['lr'] * 10000))
                print('Tmax: {:.2f}, Tmin: {:.2f}'.format(threshold_matrix.max().item(), threshold_matrix.min().item()))
            if  i % 100 == 0:
                box_pre, boxes = self.get_boxInfo_from_Binar_map(binar_map[0].detach().cpu().numpy())
                if not os.path.exists(self.args.vis_dir):
                    os.makedirs(self.args.vis_dir)
                vis_results(self.args.vis_dir, 0, self.restore_transform, img, pre_map[0].detach().cpu().numpy(), gt_map[0].detach().cpu().numpy(),
                            binar_map.detach().cpu().numpy(), threshold_matrix.detach().cpu().numpy(),boxes)

    def get_boxInfo_from_Binar_map(self, Binar_numpy, min_area=3):
        Binar_numpy = Binar_numpy.squeeze().astype(np.uint8)
        assert Binar_numpy.ndim == 2
        cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(Binar_numpy, connectivity=4)
        boxes = stats[1:, :]
        points = centroids[1:, :]
        index = (boxes[:, 4] >= min_area)
        boxes = boxes[index]
        points = points[index]
        pre_data = {'num': len(points), 'points': points}
        return pre_data, boxes

    def validate(self):
        self.net.eval()
        num_classes = 6
        losses = AverageMeter()
        cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter(), 'nae': AverageMeter()}
        metrics_s = {'tp': AverageMeter(), 'fp': AverageMeter(), 'fn': AverageMeter(), 'tp_c': AverageCategoryMeter(num_classes), 'fn_c': AverageCategoryMeter(num_classes)}
        metrics_l = {'tp': AverageMeter(), 'fp': AverageMeter(), 'fn': AverageMeter(), 'tp_c': AverageCategoryMeter(num_classes), 'fn_c': AverageCategoryMeter(num_classes)}
        gen_tqdm = tqdm(self.val_loader)
        for vi, data in enumerate(gen_tqdm, 0):
            img, dot_map, gt_data = data # [1, 3, 848, 1280], [1, 1, 848, 1280]
            slice_h, slice_w = self.cfg_data.TRAIN_SIZE
            with torch.no_grad():
                img = Variable(img).cuda()
                dot_map = Variable(dot_map).cuda()
                crop_imgs, crop_gt, crop_masks = [], [], []
                b, c, h, w = img.shape
                if h * w < slice_h * 2 * slice_w * 2 and h % 16 == 0 and w % 16 == 0:
                    [pred_threshold, pred_map, __] = [i.cpu() for i in self.net(img, mask_gt=None, mode = 'val')]
                else:
                    if h % 16 != 0:
                        pad_dims = (0, 0, 0, 16 - h % 16)
                        h = (h // 16 + 1) * 16
                        img = F.pad(img, pad_dims, "constant")
                        dot_map = F.pad(dot_map, pad_dims, "constant")
                    if w % 16 != 0:
                        pad_dims = (0, 16 - w % 16, 0, 0)
                        w =  (w // 16 + 1) * 16
                        img = F.pad(img, pad_dims, "constant")
                        dot_map = F.pad(dot_map, pad_dims, "constant")
                    assert img.size()[2:] == dot_map.size()[2:]
                    for i in range(0, h, slice_h):
                        h_start, h_end = max(min(h - slice_h, i), 0), min(h, i + slice_h)
                        for j in range(0, w, slice_w):
                            w_start, w_end = max(min(w - slice_w, j), 0), min(w, j + slice_w)
                            crop_imgs.append(img[:, :, h_start:h_end, w_start:w_end])
                            crop_gt.append(dot_map[:, :, h_start:h_end, w_start:w_end])
                            mask = torch.zeros_like(dot_map).cpu()
                            mask[:, :,h_start:h_end, w_start:w_end].fill_(1.0)
                            crop_masks.append(mask)
                    crop_imgs, crop_gt, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_imgs, crop_gt, crop_masks))
                    crop_preds, crop_thresholds = [], []
                    nz, period = crop_imgs.size(0), self.cfg_data.TRAIN_BATCH_SIZE
                    for i in range(0, nz, period):
                        [crop_threshold, crop_pred, __] = [i.cpu() for i in self.net(crop_imgs[i:min(nz, i+period)],mask_gt = None, mode='val')]
                        crop_preds.append(crop_pred)
                        crop_thresholds.append(crop_threshold)
                    crop_preds = torch.cat(crop_preds, dim=0)
                    crop_thresholds = torch.cat(crop_thresholds, dim=0)
                    idx = 0
                    pred_map = torch.zeros_like(dot_map).cpu().float()
                    pred_threshold = torch.zeros_like(dot_map).cpu().float()
                    for i in range(0, h, slice_h):
                        h_start, h_end = max(min(h - slice_h, i), 0), min(h, i + slice_h)
                        for j in range(0, w, slice_w):
                            w_start, w_end = max(min(w - slice_w, j), 0), min(w, j + slice_w)
                            pred_map[:, :, h_start:h_end, w_start:w_end]  += crop_preds[idx]
                            pred_threshold[:, :, h_start:h_end, w_start:w_end] += crop_thresholds[idx]
                            idx += 1
                    mask = crop_masks.sum(dim=0)
                    pred_map = (pred_map / mask)
                    pred_threshold = (pred_threshold / mask)
                a = torch.ones_like(pred_map)
                b = torch.zeros_like(pred_map)
                binar_map = torch.where(pred_map >= pred_threshold, a, b)
                dot_map = dot_map.cpu()
                loss = F.mse_loss(pred_map, dot_map)
                losses.update(loss.item())
                binar_map = binar_map.numpy()
                pred_data,boxes = self.get_boxInfo_from_Binar_map(binar_map)
                tp_s, fp_s, fn_s, tp_c_s, fn_c_s, tp_l, fp_l, fn_l, tp_c_l, fn_c_l = eval_metrics(num_classes,pred_data,gt_data)
                metrics_s['tp'].update(tp_s)
                metrics_s['fp'].update(fp_s)
                metrics_s['fn'].update(fn_s)
                metrics_s['tp_c'].update(tp_c_s)
                metrics_s['fn_c'].update(fn_c_s)
                metrics_l['tp'].update(tp_l)
                metrics_l['fp'].update(fp_l)
                metrics_l['fn'].update(fn_l)
                metrics_l['tp_c'].update(tp_c_l)
                metrics_l['fn_c'].update(fn_c_l)
                gt_count, pred_cnt = gt_data['num'].numpy().astype(float), pred_data['num']
                s_mae = abs(gt_count - pred_cnt)
                s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))
                cnt_errors['mae'].update(s_mae)
                cnt_errors['mse'].update(s_mse)
                if gt_count != 0:
                    s_nae = (abs(gt_count - pred_cnt) / gt_count)
                    cnt_errors['nae'].update(s_nae)
                if vi == 0:
                    vis_results(self.args.vis_dir, self.epoch, self.restore_transform, img, pred_map.numpy(), dot_map.numpy(),binar_map, pred_threshold.numpy(),boxes)
        ap_s = metrics_s['tp'].sum / (metrics_s['tp'].sum + metrics_s['fp'].sum + 1e-20)
        ar_s = metrics_s['tp'].sum / (metrics_s['tp'].sum + metrics_s['fn'].sum + 1e-20)
        f1m_s = 2 * ap_s * ar_s / (ap_s + ar_s + 1e-20 )
        ar_c_s = metrics_s['tp_c'].sum / (metrics_s['tp_c'].sum + metrics_s['fn_c'].sum + 1e-20)
        ap_l = metrics_l['tp'].sum / (metrics_l['tp'].sum + metrics_l['fp'].sum + 1e-20)
        ar_l = metrics_l['tp'].sum / (metrics_l['tp'].sum + metrics_l['fn'].sum + 1e-20)
        f1m_l = 2 * ap_l * ar_l / (ap_l + ar_l+ 1e-20)
        ar_c_l = metrics_l['tp_c'].sum / (metrics_l['tp_c'].sum + metrics_l['fn_c'].sum + 1e-20)
        loss = losses.avg
        mae = cnt_errors['mae'].avg
        mse = np.sqrt(cnt_errors['mse'].avg)
        nae = cnt_errors['nae'].avg
        self.train_record = update_model(self, [f1m_l, ap_l, ar_l,mae, mse, nae, loss], self.args)
        print_NWPU_summary(self,[f1m_l, ap_l, ar_l, mae, mse, nae, loss])