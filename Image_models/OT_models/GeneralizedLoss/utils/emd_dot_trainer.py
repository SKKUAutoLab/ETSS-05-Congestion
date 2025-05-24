import matplotlib
matplotlib.use('Agg')
from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import torch
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import logging
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.vgg import vgg19
from datasets.crowd import Crowd
from geomloss import SamplesLoss

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

def grid(H, W, stride):
    coodx = torch.arange(0, W, step=stride) + stride / 2
    coody = torch.arange(0, H, step=stride) + stride / 2
    y, x = torch.meshgrid( [  coody.type(dtype) / 1, coodx.type(dtype) / 1 ] )
    return torch.stack( (x,y), dim=2 ).view(-1,2)

def per_cost(X, Y):
    x_col = X.unsqueeze(-2)
    y_lin = Y.unsqueeze(-3)
    C = torch.sum((torch.abs(x_col - y_lin)) ** 2, -1)
    C = torch.sqrt(C)
    s = (x_col[:,:,:,-1] + y_lin[:,:,:,-1]) / 2
    s = s * 0.2 + 0.5
    return (torch.exp(C/s) - 1)

def exp_cost(X, Y):
    x_col = X.unsqueeze(-2)
    y_lin = Y.unsqueeze(-3)
    C = torch.sum((torch.abs(x_col - y_lin)) ** 2, -1)
    C = torch.sqrt(C)
    return (torch.exp(C/scale) - 1.)

def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, targets, st_sizes

class EMDTrainer(Trainer):
    def setup(self):
        args = self.args
        global scale
        scale = args.scale
        if args.cost == 'exp':
            self.cost = exp_cost
        elif args.cost == 'per':
            self.cost = per_cost
        else:
            print('This cost function does not exist')
            raise NotImplementedError
        self.downsample_ratio = args.downsample_ratio
        # train and test loader (1 is num gpus)
        self.datasets = {x: Crowd(os.path.join(args.input_dir, x), args.crop_size, args.downsample_ratio, args.is_gray, x) for x in ['train', 'val']}
        self.dataloaders = {x: DataLoader(self.datasets[x], collate_fn=(train_collate if x == 'train' else default_collate), batch_size=(self.args.batch_size if x == 'train' else 1), shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers * 1, pin_memory=(True if x == 'train' else False), drop_last=True) for x in ['train', 'val']}
        # model
        self.model = vgg19()
        self.model.cuda()
        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.start_epoch = 0
        # resume training
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, map_location='cuda')
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, map_location='cuda'))
            print('Load ckpt from:', args.resume)
        # loss
        self.blur = args.blur
        self.criterion = SamplesLoss(blur=args.blur, scaling=args.scaling, debias=False, backend='tensorized', cost=self.cost, reach=args.reach, p=args.p)
        self.writer = SummaryWriter(args.output_dir)
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_mae_list = []
        self.best_mae = {}
        self.best_mse = {}
        self.best_epoch = {}
        for stage in ['train', 'val']:
            self.best_mae[stage] = np.inf
            self.best_mse[stage] = np.inf
            self.best_epoch[stage] = 0

    def train(self):
        args = self.args
        for epoch in range(self.start_epoch, args.epochs):
            self.epoch = epoch
            self.train_eopch(epoch)
            # if epoch % args.val_epoch == 0 and epoch >= args.val_start:
            if epoch % 1 == 0:
                self.val_epoch(stage='val')

    def train_eopch(self, epoch=0):
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        self.model.train()
        if epoch < 10:
            for param_group in self.optimizer.param_groups:
                if param_group['lr'] >= 0.1*self.args.lr:
                    param_group['lr'] = self.args.lr * (epoch + 1) / 10
        for step, (inputs, points, targets, st_sizes) in enumerate(self.dataloaders['train']):
            inputs = inputs.cuda() # [1, 3, 512, 512]
            gd_count = np.array([len(p) for p in points], dtype=np.float32) # [1]
            points = [p.cuda() for p in points]
            shape = (inputs.shape[0],int(inputs.shape[2]/self.args.downsample_ratio),int(inputs.shape[3]/self.args.downsample_ratio)) # (1, 64, 64)
            with torch.autograd.set_grad_enabled(True):
                outputs = self.model(inputs) # [1, 1, 64, 64]
                cood_grid = grid(outputs.shape[2], outputs.shape[3], 1).unsqueeze(0) * self.args.downsample_ratio + (self.args.downsample_ratio / 2) # [1, 4096, 2]
                cood_grid = cood_grid.type(torch.cuda.FloatTensor) / float(self.args.crop_size) # [1, 4096, 2]
                i = 0
                emd_loss = 0
                point_loss = 0
                pixel_loss = 0
                entropy = 0
                for p in points:
                    if len(p) < 1:
                        gt = torch.zeros((1, shape[1], shape[2])).cuda() # [1, 64, 64]
                        point_loss += torch.abs(gt.sum() - outputs[i].sum()) / shape[0]
                        pixel_loss += torch.abs(gt.sum() - outputs[i].sum()) / shape[0]
                        emd_loss += torch.abs(gt.sum() - outputs[i].sum()) / shape[0]
                    else:
                        gt = torch.ones((1, len(p), 1)).cuda() # [1, 159, 1]
                        cood_points = p.reshape(1, -1, 2) / float(self.args.crop_size) # [1, 159, 2]
                        A = outputs[i].reshape(1, -1, 1) # [1, 4096, 1]
                        l, F, G = self.criterion(A, cood_grid, gt, cood_points)
                        C = self.cost(cood_grid, cood_points) # [1, 4096, 159]
                        PI = torch.exp((F.repeat(1,1,C.shape[2]) + G.permute(0,2,1).repeat(1, C.shape[1], 1) - C).detach() / self.args.blur**self.args.p) * A * gt.permute(0, 2, 1) # [1, 4096, 159]
                        entropy += torch.mean((1e-20 + PI) * torch.log(1e-20 + PI)) # [1]
                        emd_loss += (torch.mean(l) / shape[0])
                        if self.args.d_point == 'l1':
                            point_loss += torch.sum(torch.abs(PI.sum(1).reshape(1,-1,1)-gt)) / shape[0] 
                        else:
                            point_loss += torch.sum((PI.sum(1).reshape(1,-1,1)-gt)**2) / shape[0] 
                        if self.args.d_pixel == 'l1':
                            pixel_loss += torch.sum(torch.abs(PI.sum(2).reshape(1,-1,1).detach()-A)) / shape[0] 
                        else:
                            pixel_loss += torch.sum((PI.sum(2).reshape(1,-1,1).detach()-A)**2) / shape[0] 
                    i += 1
                loss = emd_loss + self.args.tau*(pixel_loss + point_loss) + self.blur*entropy
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                N = inputs.size(0)
                outputs = torch.mean(outputs, dim=1) # [1, 64, 64]
                pre_count = torch.sum(outputs[-1]).detach().cpu().numpy()
                res = (pre_count - gd_count[-1])
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)
        logging.info('Epoch: {}, Loss: {:.4f}, MSE: {:.4f}, MAE: {:.4f}'.format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg()))
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({'epoch': self.epoch, 'optimizer_state_dict': self.optimizer.state_dict(), 'model_state_dict': model_state_dic}, save_path)
        self.save_list.append(save_path)

    def val_epoch(self, stage='val'):
        self.model.eval()
        epoch_res = []
        if stage == 'val':
            dataloader = self.dataloaders['val']
        for inputs, points, name in dataloader:
            inputs = inputs.cuda() # [1, 3, 1875, 2500]
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs) # [1, 1, 234, 312]
                points = points[0].type(torch.LongTensor) # [506, 2]
                res = len(points) - torch.sum(outputs).item()
                epoch_res.append(res)
        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        self.writer.add_scalar(stage + '/mae', mae, self.epoch)
        self.writer.add_scalar(stage + '/mse', mse, self.epoch)
        logging.info('{}: Epoch {}, MSE: {:.4f}, MAE: {:.4f}'.format(stage, self.epoch, mse, mae))
        model_state_dic = self.model.state_dict()
        if mae < self.best_mae[stage]:
            self.best_mse[stage] = mse
            self.best_mae[stage] = mae
            self.best_epoch[stage] = self.epoch 
            logging.info("{}: Save best MSE: {:.4f}, MAE: {:.4f} at epoch {}".format(stage, self.best_mse[stage], self.best_mae[stage], self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_{}.pth').format(stage))
        logging.info('Val: Best epoch {}, MSE: {:.4f}, MAE: {:.4f}'.format(self.best_epoch['val'], self.best_mse['val'], self.best_mae['val']))