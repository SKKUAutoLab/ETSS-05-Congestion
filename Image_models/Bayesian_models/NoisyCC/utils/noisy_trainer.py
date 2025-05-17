from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import logging
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.vgg import vgg19
from datasets.crowd import Crowd
from losses.full_post_prob import Full_Post_Prob
from losses.full_covar import Full_Cov_Gaussian_Loss 

def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, targets, st_sizes

class NoisyTrainer(Trainer):
    def setup(self):
        args = self.args
        self.loss = args.loss
        self.skip_test = args.skip_test
        self.add = args.add
        self.downsample_ratio = args.downsample_ratio
        lists = {}
        train_list = None
        val_list = None
        test_list = None
        lists['train'] = train_list
        lists['val'] = val_list
        lists['test'] = test_list
        # train and test loader (1 is num gpus)
        self.datasets = {x: Crowd(os.path.join(args.input_dir, x), args.crop_size, args.downsample_ratio, args.is_gray, x,  im_list=lists[x]) for x in ['train', 'val']}
        self.dataloaders = {x: DataLoader(self.datasets[x], collate_fn=(train_collate if x == 'train' else default_collate), batch_size=(self.args.batch_size if x == 'train' else 1), shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers * 1, pin_memory=(True if x == 'train' else False)) for x in ['train', 'val']}
        self.datasets['test'] = Crowd(os.path.join(args.input_dir, 'test'), args.crop_size, args.downsample_ratio, args.is_gray, 'val', im_list=lists['test'])
        self.dataloaders['test'] = DataLoader(self.datasets['test'], collate_fn=default_collate, batch_size=1, shuffle=False, num_workers=args.num_workers * 1, pin_memory=False)
        # model
        self.model = vgg19(down=self.downsample_ratio, bn=args.bn, o_cn=args.o_cn)
        self.model.cuda()
        # optimizer
        params = list(self.model.parameters()) 
        self.optimizer = optim.Adam(params, lr=args.lr)
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
            print('Load ckpt from {}'.format(args.resume))
        # loss
        self.post_prob = Full_Post_Prob(args.sigma, args.alpha, args.crop_size, args.downsample_ratio, args.background_ratio, args.use_background, add=self.add, minx=args.minx, ratio=args.ratio)
        self.criterion = Full_Cov_Gaussian_Loss(args.use_background, weight=self.args.weight, reg=args.reg)
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.test_flag = False
        self.best_mae = {}
        self.best_mse = {}
        self.best_epoch = {}
        for stage in ['val', 'test']:
            self.best_mae[stage] = np.inf
            self.best_mse[stage] = np.inf
            self.best_epoch[stage] = 0

    def train(self):
        args = self.args
        for epoch in range(self.start_epoch, args.epochs):
            self.epoch = epoch
            self.train_eopch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()
                if self.test_flag and not self.skip_test:
                    self.val_epoch(stage='test')
                    self.test_flag = False

    def train_eopch(self):
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        self.model.train()
        self.optimizer.zero_grad()
        for step, (inputs, points, targets, st_sizes) in enumerate(self.dataloaders['train']):
            inputs = inputs.cuda() # [1, 3, 512, 512]
            st_sizes = st_sizes.cuda() # [1]
            gd_count = np.array([len(p) for p in points], dtype=np.float32) # [1]
            points = [p.cuda() for p in points]
            targets = [t.cuda() for t in targets]
            with torch.autograd.set_grad_enabled(True):
                outputs = self.model(inputs) # [1, 1, 64, 64]
                prob_list = self.post_prob(points, st_sizes)
                loss = self.criterion(prob_list, targets, outputs)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                N = inputs.size(0)
                pre_count = torch.sum(outputs[0]).detach().cpu().numpy()
                res = (pre_count - gd_count[0])
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)
        logging.info('Epoch: {}, Loss: {:.4f}, MSE: {:.4f}, MAE: {:.4f}'.format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg()))
        # save model
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({'epoch': self.epoch, 'optimizer_state_dict': self.optimizer.state_dict(), 'model_state_dict': model_state_dic}, save_path)
        self.save_list.append(save_path)

    def val_epoch(self, stage='val'):
        self.model.eval()
        epoch_res = []
        for inputs, points, name in self.dataloaders[stage]:
            inputs = inputs.cuda() # [1, 3, 1875, 2000]
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs) # [1, 1, 234, 312]
                points = points[0].type(torch.LongTensor) # [506, 2]
                res = len(points) - torch.sum(outputs).item()
                epoch_res.append(res)
        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        logging.info('{}: Epoch: {}, MSE: {:.4f}, MAE: {:.4f}'.format(stage, self.epoch, mse, mae))
        model_state_dic = self.model.state_dict()
        if (2.0 * mse + mae) < (2.0 * self.best_mse[stage] + self.best_mae[stage]):
            self.test_flag = True
            self.best_mse[stage] = mse
            self.best_mae[stage] = mae
            self.best_epoch[stage] = self.epoch 
            logging.info("{}: Save best MSE {:.4f}, MAE: {:.4f} at epoch {}".format(stage, self.best_mse[stage], self.best_mae[stage], self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_{}.pth').format(stage))
        logging.info('Val: Best Epoch: {} MSE: {:.4f}, MAE: {:.2f}'.format(self.best_epoch['val'], self.best_mse['val'], self.best_mae['val']))
        logging.info('Test: Best Epoch: {}, MSE: {:.4f}, MAE: {:.4f}'.format(self.best_epoch['test'], self.best_mse['test'], self.best_mae['test']))