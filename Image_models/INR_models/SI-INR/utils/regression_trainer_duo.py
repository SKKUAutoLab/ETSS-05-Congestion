from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.SI_INR_new3 import New_bay_Net
from datasets.crowd import Crowd
from losses.bay_loss import Bay_Loss
from losses.post_prob_duo import Post_Prob
import random
from torch.optim import lr_scheduler
import cv2
from matplotlib import pyplot as plt
import torchvision.transforms.functional as F

def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    grid_c = torch.stack(transposed_batch[4], 0)
    gridnum_sam_c = transposed_batch[5]
    gd_count = transposed_batch[6]
    return images, points, targets, st_sizes, grid_c, gridnum_sam_c, gd_count

class RegTrainer(Trainer):
    def setup(self):
        args = self.args
        self.downsample_ratio = args.downsample_ratio
        # train and test loader, 1 is device_count
        self.datasets = {x: Crowd((os.path.join(args.input_dir, 'train_data/images') if x == 'train' else os.path.join(args.input_dir, 'test_data/images')),
                                  args.crop_size, args.downsample_ratio, args.is_gray, x) for x in ['train', 'val']}
        g = torch.Generator()
        g.manual_seed(args.seed)
        self.dataloaders = {x: DataLoader(self.datasets[x], collate_fn=(train_collate if x == 'train' else default_collate), batch_size=(args.batch_size if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False), num_workers=args.num_workers * 1, pin_memory=(True if x == 'train' else False)) for x in ['train', 'val']}
        # model
        self.model = New_bay_Net()
        self.model.cuda()
        # optimizer
        c_params = list(map(id, self.model.modelA.parameters()))
        b_params = filter(lambda p: id(p) not in c_params, self.model.parameters())
        self.optimizer1 = optim.Adam(b_params, lr=args.lr, weight_decay=args.weight_decay)
        self.optimizer2 = optim.Adam(self.model.modelA.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = lr_scheduler.StepLR(self.optimizer1, step_size = 3000, gamma = 1)
        self.start_epoch = 0
        # resume training
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, 'cuda')
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer1.load_state_dict(checkpoint['optimizer_state_dict1'])
                self.optimizer2.load_state_dict(checkpoint['optimizer_state_dict2'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, 'cuda'))
        # loss
        self.post_prob = Post_Prob(args.sigma, args.crop_size, args.downsample_ratio, args.background_ratio, args.use_background)
        self.criterion = Bay_Loss(args.use_background)
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_mae = np.inf
        self.best_mse = np.inf

    def train(self):
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            self.epoch = epoch
            self.train_epoch()
            self.scheduler.step()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()

    def train_epoch(self):
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        self.model.train()
        # [4, 3, 256, 256], [4], [4], [4], [4, 1024, 2], [4], [4]
        for step, (inputs, points, targets, st_sizes, grid_c, gridnum_sam_c, gd_count) in enumerate(self.dataloaders['train']):
            _scale = 1 + (random.random() - 0.5) * 0.8 - 0.3
            _h = round(512 * _scale)
            _w = round(512 * _scale)
            inputs = inputs.cuda() # [4, 3, 256, 256]
            points = [p.cuda() for p in points] # [4]
            targets = [t.cuda() for t in targets] # [4]
            st_sizes = st_sizes.cuda() # [4]
            gridnum_sam_c = [tt.cuda() for tt in gridnum_sam_c] # [4]
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs) # [4, 1, 64, 64]
                prob_list = self.post_prob(points, st_sizes, gridnum_sam_c)
                loss1 = self.criterion(prob_list, targets, outputs)
                loss_KL = self.model.kl_div
                outputs = outputs / 10
                Train_uq = False
                if Train_uq:
                    loss = 1.0 * loss1 + 0.1 * loss_KL
                    self.optimizer1.zero_grad()
                    self.optimizer2.zero_grad()
                    loss.backward()
                    self.optimizer2.step()
                else:
                  loss = 1.0 * loss1
                  self.optimizer1.zero_grad()
                  self.optimizer2.zero_grad()
                  loss.backward()
                  self.optimizer1.step()
                  self.optimizer2.step()
                N = inputs.size(0)
                pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                res = pre_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)
        print('[Train]: Epoch: [{}/{}], Loss: {:.4f}, MSE: {:.2f}, MAE: {:.2f}'.format(self.epoch + 1, self.args.max_epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg()))
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({'epoch': self.epoch, 'optimizer_state_dict1': self.optimizer1.state_dict(), 'optimizer_state_dict2': self.optimizer2.state_dict(),
                    'model_state_dict': model_state_dic}, save_path)
        self.save_list.append(save_path)

    def val_epoch(self):
        self.model.eval()
        epoch_res = []
        c_results = []
        scale_ls = [1.0]
        for inputs, count, name, cor_C in self.dataloaders['val']:
            c_sub_result = []
            epoch_sub_res = []
            for i in range(len(scale_ls)):
                scale = scale_ls[i]
                _w = round(512 * scale)
                inputs_ = F.resize(inputs, _w) # [1, 3, 512, 512]
                inputs_ = inputs_.cuda()
                inputs = inputs.cuda() # [1, 3, 256, 256]
                assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
                with torch.set_grad_enabled(False):
                    outputs = self.model(inputs_) # [1, 1, 64, 64], [4096]
                    outputs = outputs / 10
                    res = count[0].item() - torch.sum(outputs).item()
                    epoch_sub_res.append(res)
                    c_sub_result.append(outputs.data.cpu().numpy())
            c_results.append(c_sub_result) 
            epoch_res.append(epoch_sub_res)
        epoch_res = np.array(epoch_res) # [182, 1]
        mses = np.sqrt(np.mean(np.square(epoch_res), axis = 0, keepdims=True))
        maes = np.mean(np.abs(epoch_res),axis = 0, keepdims=True)
        mae = np.mean(maes)
        mse = np.mean(mses)
        print('[Val]: Epoch: [{}/{}], MSE: {:.2f}, MAE: {:.2f}'.format(self.epoch + 1, self.args.max_epoch, mse, mae))
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            print("Save best MSE: {:.2f}, Best MAE: {:.2f} at epoch: {}".format(self.best_mse, self.best_mae, self.epoch + 1))
            torch.save(self.model, os.path.join(self.save_dir, 'best_model.pt'))
        if abs(mae - self.best_mae) < 10:
            torch.save(self.model, os.path.join(self.save_dir, 'recent_model.pt'))
        if abs(mae - self.best_mae) < 5:
            torch.save(self.model, os.path.join(self.save_dir, 'fine_model.pt'))
        if self.epoch % 10 == 0:
            fig = plt.figure()
            rows = 4
            columns = len(scale_ls)
            for i in range(4):
                for j in range(len(scale_ls)):
                    c_example = c_results[i][j]
                    c_example = cv2.resize(c_example[0][0], (64, 64), interpolation=cv2.INTER_CUBIC)
                    fig.add_subplot(rows, columns, i * len(scale_ls) + 1 + j)
                    plt.imshow(c_example)
            plt.savefig(os.path.join(self.save_dir, str(self.epoch) + '_test_fig.png'))
            plt.close()