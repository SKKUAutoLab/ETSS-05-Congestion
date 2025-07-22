import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from trainers.trainer import Trainer
from utils.misc import denormalize, divide_img_into_patches
import matplotlib
matplotlib.use('Agg') # fix main thread is not in main loop error

class DGTrainer(Trainer):
    def __init__(self, seed, version, log_para, patch_size, mode, rank, device, is_ddp):
        super().__init__(seed, version, rank, device, is_ddp)
        self.log_para = log_para
        self.patch_size = patch_size
        self.mode = mode
        self.device = device

    def load_ckpt(self, model, path):
        super().load_ckpt(model, path)

    def save_ckpt(self, model, path):
        super().save_ckpt(model, path)

    def compute_count_loss(self, loss: nn.Module, pred_dmaps, gt_datas, weights=None):
        if loss.__class__.__name__ == 'MSELoss':
            _, gt_dmaps, _ = gt_datas # [16, 1, 320, 320]
            gt_dmaps = gt_dmaps.to(self.device)
            if weights is not None:
                pred_dmaps = pred_dmaps * weights
                gt_dmaps = gt_dmaps * weights
            loss_value = loss(pred_dmaps, gt_dmaps * self.log_para)
        elif loss.__class__.__name__ == 'BL':
            gts, targs, st_sizes = gt_datas
            gts = [gt.to(self.device) for gt in gts]
            targs = [targ.to(self.device) for targ in targs]
            st_sizes = st_sizes.to(self.device)
            loss_value = loss(gts, st_sizes, targs, pred_dmaps)
        else:
            print('This loss does not exist')
            raise NotImplementedError
        return loss_value

    def predict(self, model, img): # [1, 3, 768, 1024]
        h, w = img.shape[2:]
        ps = self.patch_size
        if h >= ps or w >= ps:
            pred_count = 0
            img_patches, _, _ = divide_img_into_patches(img, ps)
            for patch in img_patches:
                pred = model(patch, train=False)[0]
                pred_count += torch.sum(pred).cpu().item() / self.log_para
        else:
            pred_dmap = model(img, train=False)[0] # [1, 1, 768, 1024]
            pred_count = pred_dmap.sum().cpu().item() / self.log_para
        return pred_count
    
    def get_visualized_results(self, model, img):
        h, w = img.shape[2:]
        ps = self.patch_size
        if h >= ps or w >= ps:
            dmap = torch.zeros(1, 1, h, w)
            img_patches, nh, nw = divide_img_into_patches(img, ps)
            for i in range(nh):
                for j in range(nw):
                    patch = img_patches[i * nw + j]
                    pred_dmap = model(patch, train=False)
                    dmap[:, :, i * ps:(i + 1) * ps, j * ps:(j + 1) * ps] = pred_dmap
        else:
            dmap = model(img, train=False)
        dmap = dmap[0, 0].cpu().detach().numpy().squeeze()
        return dmap
    
    def get_visualized_results_with_cls(self, model, img):
        h, w = img.shape[2:]
        ps = self.patch_size
        if h >= ps or w >= ps:
            dmap = torch.zeros(1, 1, h, w)
            cmap = torch.zeros(1, 3, h // 16, w // 16)
            img_patches, nh, nw = divide_img_into_patches(img, ps)
            for i in range(nh):
                for j in range(nw):
                    patch = img_patches[i * nw + j]
                    pred_dmap, pred_cmap = model(patch, train=False)
                    dmap[:, :, i * ps:(i + 1) * ps, j * ps:(j + 1) * ps] = pred_dmap
                    cmap[:, :, i * ps // 16:(i + 1) * ps // 16, j * ps // 16:(j + 1) * ps // 16] = pred_cmap
        else:
            dmap, cmap = model(img, train=False)
        dmap = dmap[0, 0].cpu().detach().numpy().squeeze()
        cmap = cmap[0, 0].cpu().detach().numpy().squeeze()
        return dmap, cmap
    
    def train_step(self, model, loss, optimizer, batch, epoch):
        imgs1, imgs2, gt_datas = batch # [16, 3, 320, 320], [16, 3, 320, 320], ([76, 2] * len(16), [16, 1, 320, 320], [16, 1, 20, 20])
        imgs1 = imgs1.to(self.device)
        imgs2 = imgs2.to(self.device)
        gt_cmaps = gt_datas[-1].to(self.device) # [16, 1, 20, 20]
        if self.mode == 'final':
            optimizer.zero_grad()
            # [16, 1, 320, 320], [16, 1, 320, 320], [16, 1, 20, 20], [16, 1, 20, 20], [16, 1, 320, 320], [1], 0
            dmaps1, dmaps2, cmaps1, cmaps2, cerrmap, loss_con, loss_err = model(imgs1, imgs2, gt_cmaps, train=True)
            loss_den = self.compute_count_loss(loss, dmaps1, gt_datas) + self.compute_count_loss(loss, dmaps2, gt_datas)
            loss_cls = F.binary_cross_entropy(cmaps1, gt_cmaps) + F.binary_cross_entropy(cmaps2, gt_cmaps)
            loss_total = loss_den + 10 * loss_cls + 10 * loss_con # + loss_err
            loss_total.backward()
            optimizer.step()
        else:
            print('This model mode does not exist')
            raise NotImplementedError
        return loss_total.detach().item()

    def val_step(self, model, batch):
        img1, img2, gt, _, _ = batch # [1, 3, 768, 1024], [1, 3, 768, 1024], [1, 298, 2]
        img1 = img1.to(self.device)
        if self.mode == 'final':
            pred_count = self.predict(model, img1)
        else:
            print('This model mode does not exist')
            raise NotImplementedError
        gt_count = gt.shape[1]
        mae = np.abs(pred_count - gt_count)
        mse = (pred_count - gt_count)**2
        return mae, {'mse': mse}

    def test_step(self, model, batch):
        img1, _, gt, _, _ = batch # [1, 3, 1888, 2512], [1, 975, 2]
        img1 = img1.to(self.device)
        if self.mode == 'final':
            pred_count = self.predict(model, img1)
        else:
            print('This model mode does not exist')
            raise NotImplementedError
        gt_count = gt.shape[1]
        mae = np.abs(pred_count - gt_count)
        mse = (pred_count - gt_count)**2
        return {'mae': mae, 'mse': mse}
        
    def vis_step(self, model, batch):
        img1, img2, gt, name, _ = batch # [1, 3, 768, 1024], [1, 3, 768, 1024], [1, 23, 2], ['IMG_1']
        vis_dir = os.path.join(self.log_dir, 'vis')
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        if self.mode == 'final':
            pred_dmap1, pred_cmap1 = self.get_visualized_results_with_cls(model, img1) # [768, 1024], [48, 64]
            pred_dmap2, pred_cmap2 = self.get_visualized_results_with_cls(model, img2) # [768, 1024], [48, 64]
            img1 = denormalize(img1.detach())[0].cpu().permute(1, 2, 0).numpy() # [768, 1024, 3]
            img2 = denormalize(img2.detach())[0].cpu().permute(1, 2, 0).numpy() # [768, 1024, 3]
            pred_count1 = pred_dmap1.sum() / self.log_para
            pred_count2 = pred_dmap2.sum() / self.log_para
            gt_count = gt.shape[1]
            new_cmap1 = pred_cmap1.copy() # [48, 64]
            new_cmap1[new_cmap1 < 0.5] = 0
            new_cmap1[new_cmap1 >= 0.5] = 1
            new_cmap2 = pred_cmap2.copy() # [48, 64]
            new_cmap2[new_cmap2 < 0.5] = 0
            new_cmap2[new_cmap2 >= 0.5] = 1
            datas = [img1, pred_dmap1, pred_cmap1, img2, pred_dmap2, pred_cmap2]
            titles = [name[0], f'Pred1: {pred_count1}', 'Cls1', f'GT: {gt_count}', f'Pred2: {pred_count2}', 'Cls2']
            fig = plt.figure(figsize=(15, 6))
            for i in range(6):
                ax = fig.add_subplot(2, 3, i + 1)
                ax.set_title(titles[i])
                ax.imshow(datas[i])
            plt.savefig(os.path.join(vis_dir, f'{name[0]}.png'))
            new_datas = [img1, pred_cmap1, new_cmap1, pred_dmap1]
            new_titles = [f'{name[0]}', 'Cls', 'BCls', f'Pred_{pred_count1}']
            for i in range(len(new_datas)):
                plt.imsave(os.path.join(vis_dir, f'{name[0]}_{new_titles[i]}.png'), new_datas[i])
            plt.close()
        else:
            print('This model mode does not exist')
            raise NotImplementedError