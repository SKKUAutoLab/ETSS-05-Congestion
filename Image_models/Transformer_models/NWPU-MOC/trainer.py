import numpy as np
import torch
from torch import optim
from models.CC import CrowdCounter
from config import cfg
from misc.utils import save_checkpoint, is_validation, AverageMeter, AverageCategoryMeter, eval_mc, update_model
import datasets
from misc.utils import adjust_learning_rate, adjust_double_learning_rate

class Trainer():
    def __init__(self, cfg_data, pwd):
        self.cfg_data = cfg_data
        self.train_loader, self.val_loader = datasets.loading_data(cfg.DATASET)
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd
        self.net_name = cfg.NET
        self.gpu_id = cfg.GPU_ID
        # model
        self.net = CrowdCounter(self.net_name, self.gpu_id)
        self.categorys = self.cfg_data.CATEGORYS
        self.num_classes = len(self.categorys)
        # optimizer
        if cfg.BACKBONE_FREEZE:       
            self.optimizer = optim.AdamW([{'params': [param for name, param in self.net.named_parameters() if 'backbone' in name], 'lr': cfg.CONV_LR, 'weight_decay': cfg.WEIGHT_DECAY},
                                          {'params': [param for name, param in self.net.named_parameters() if 'backbone' not in name], 'lr': cfg.BASE_LR, 'weight_decay': cfg.WEIGHT_DECAY}])
        else:
            self.optimizer = optim.AdamW(self.net.parameters(), lr=cfg.BASE_LR, weight_decay=cfg.WEIGHT_DECAY)
        self.train_record = {'best_cls_avg_mae': 1e20, 'best_cls_avg_mse':1e20, 'best_cls_weight_mse':1e20, 'best_model_name': ''}
        self.epoch = 0
        self.i_tb = 0
        self.num_iters = cfg.MAX_EPOCH * np.int(len(self.train_loader))
        self.train_loss_avg = 0
        self.val_loss_avg = 0
        if cfg.PRE_GCC:
            self.net.load_state_dict(torch.load(cfg.PRE_GCC_MODEL))
        if cfg.RESUME:
            latest_state = torch.load(cfg.RESUME_PATH)
            self.net.load_state_dict(latest_state['net'])
            self.optimizer.load_state_dict(latest_state['optimizer'])
            self.epoch = latest_state['epoch'] + 1
            self.i_tb = latest_state['i_tb']
            self.num_iters = latest_state['num_iters']
            self.train_record = latest_state['train_record']
            self.exp_path = latest_state['exp_path']
            print('Load ckpt from:', cfg.RESUME_PATH)

    def forward(self):
        for epoch in range(self.epoch, cfg.MAX_EPOCH):
            self.epoch = epoch
            self.train()
            if self.epoch % cfg.CP_FREQ == 0:
                save_checkpoint(self)
            self.train_loss_avg = 0
            if is_validation(cfg.VAL_STAGE,cfg.VAL_FREQ, self.epoch):
                self.validate()
                self.val_loss_avg = 0
            
    def train(self):
        self.net.train()
        train_losses = AverageMeter()
        for it, data in enumerate(self.train_loader, 0):
            self.i_tb += 1
            for k, v in data.items():
                data[k] = v.cuda()
            num_class = self.num_classes # 6
            rgb, nir, gt_map = data['rgb'], data['nir'], data['gt_map'][:, :num_class, :, :] # [4, 3, 512, 512], [4, 3, 512, 512], [4, 6, 64, 64]
            self.optimizer.zero_grad()
            outputs = self.net(data, num_class, it)
            pred_map = outputs['pred_map'] # [4, 6, 64, 64]
            gauss_map = outputs['gauss_map'] # [4, 6, 64, 64]
            loss = self.net.loss
            loss.backward()
            self.optimizer.step()
            train_losses.update(loss.item())
            if cfg.BACKBONE_FREEZE:
                base_lr, conv_lr = adjust_double_learning_rate(self.optimizer, self.i_tb, self.num_iters, base_lr=cfg.BASE_LR, conv_lr=cfg.CONV_LR)
            else:
                base_lr = adjust_learning_rate(self.optimizer, self.i_tb, self.num_iters, lr=cfg.BASE_LR)
            if (it + 1) % cfg.PRINT_FREQ == 0:
                if cfg.BACKBONE_FREEZE:
                    print('Epoch: {}, Iter: [{}/{}], Loss: {:.4f}, Base lr: {:.8f}, Conv lr: {:.8f}'.format(self.epoch + 1, it + 1, len(self.train_loader), loss.item(), base_lr, conv_lr))
                else:
                    print('Epoch: {}, Iter: [{}/{}], Loss: {:.4f}, Base lr: {:.8f}'.format(self.epoch + 1, it + 1, len(self.train_loader), loss.item(), base_lr))
                for c_idx in range(self.num_classes):
                    print('Category: {}, GT: {:.2f}, Pred: {:.2f}'.format(self.categorys[c_idx], gt_map[0][c_idx].sum().data / self.cfg_data.LOG_PARA, pred_map[0][c_idx].sum().data / self.cfg_data.LOG_PARA))
        self.train_loss_avg = train_losses.avg

    def validate(self):
        self.net.eval()
        val_losses = AverageMeter()
        maes = AverageCategoryMeter(self.num_classes)
        mses = AverageCategoryMeter(self.num_classes)
        cmses = AverageMeter()
        for index, data in enumerate(self.val_loader, 0):
            for k, v in data.items():
                data[k] = v.cuda()
            num_class = self.num_classes # 6
            rgb, nir, gt_map = data['rgb'], data['nir'], data['gt_map'][:, :num_class, :, :] # [1, 3, 1024, 1024], [1, 3, 1024, 1024], [1, 6, 128, 128]
            with torch.set_grad_enabled(False):
                if cfg.POS_EMBEDDING:
                    b, c, h, w = rgb.shape
                    rh, rw = self.cfg_data.TRAIN_SIZE
                    crop_RGBs, crop_Ts, crop_masks = [], [], []
                    for i in range(0, h, rh):
                        gis, gie = max(min(h-rh, i), 0), min(h, i + rh)
                        for j in range(0, w, rw):
                            gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                            crop_RGBs.append(rgb[:, :, gis:gie, gjs:gje])
                            crop_Ts.append(nir[:, :, gis:gie, gjs:gje])
                            mask = torch.zeros(b, 1, h // self.cfg_data.LABEL_FACTOR, w // self.cfg_data.LABEL_FACTOR).cuda()
                            mask[:, :, gis // self.cfg_data.LABEL_FACTOR:gie // self.cfg_data.LABEL_FACTOR, gjs // self.cfg_data.LABEL_FACTOR:gje // self.cfg_data.LABEL_FACTOR].fill_(1.0)
                            crop_masks.append(mask)
                    crop_RGBs, crop_Ts, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_RGBs, crop_Ts, crop_masks))
                    crop_data = {'rgb': crop_RGBs, 'nir': crop_Ts,}
                    crop_outputs = self.net(crop_data, num_class, mode='val')
                    crop_preds = crop_outputs['pred_map']
                    h, w, rh, rw = h // self.cfg_data.LABEL_FACTOR, w // self.cfg_data.LABEL_FACTOR, rh // self.cfg_data.LABEL_FACTOR, rh // self.cfg_data.LABEL_FACTOR
                    idx = 0
                    pred_map = torch.zeros(b, 6, h, w).cuda()
                    for i in range(0, h, rh):
                        gis, gie = max(min(h-rh, i), 0), min(h, i + rh)
                        for j in range(0, w, rw):
                            gjs, gje = max(min(w-rw, j), 0), min(w, j + rw)
                            pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
                            idx += 1
                    mask = crop_masks.sum(dim=0)
                    pred_map = pred_map / mask # [1, 6, 128, 128]
                else:
                    outputs = self.net(data, num_class, mode='val')
                    pred_map = outputs['pred_map'] # [1, 6, 128, 128]
                val_losses.update(self.net.loss.item())
                abs_errors, square_errors, weights = eval_mc(pred_map, gt_map, self.cfg_data.LOG_PARA)
                wmse = 0.0
                for c_idx in range(self.num_classes):
                    maes.update(abs_errors[c_idx], c_idx)
                    mses.update(square_errors[c_idx], c_idx)
                    wmse += square_errors[c_idx] * weights[c_idx]
                cmses.update(wmse)
        self.val_loss_avg = val_losses.avg
        overall_mae = maes.avg
        overall_rmse = np.sqrt(mses.avg) 
        cls_weight_mse = cmses.avg
        cls_avg_mae = sum(overall_mae) / self.num_classes
        cls_avg_rmse = sum(overall_rmse) / self.num_classes
        self.train_record = update_model(self.net, self.epoch, self.exp_path,[self.val_loss_avg, overall_mae, overall_rmse, cls_avg_mae, cls_avg_rmse, cls_weight_mse], self.train_record)
        print('Epoch: {}, Val loss: {:.4f}, cls_avg_mae: {:.2f}, cls_avg_rmse: {:.2f}, cls_weight_mse: {:.2f}'.format(self.epoch + 1, self.val_loss_avg, cls_avg_mae, cls_avg_rmse, cls_weight_mse))
        for c_idx in range(self.num_classes):
            print('Category: {}, MAE: {:.2f}, RMSE: {:.2f}'.format(self.categorys[c_idx], overall_mae[c_idx], overall_rmse[c_idx]))