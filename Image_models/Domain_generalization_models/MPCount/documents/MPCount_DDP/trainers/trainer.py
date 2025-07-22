import torch
import torch.nn as nn
import os
from glob import glob
from utils.misc import AverageMeter, DictAvgMeter, easy_track
import math
from torch.nn.parallel import DistributedDataParallel as DDP

class Trainer(object):
    def __init__(self, seed, version, rank, device, is_ddp):
        self.seed = seed
        self.version = version
        self.log_dir = os.path.join('logs', self.version)
        os.makedirs(self.log_dir, exist_ok=True)
        self.rank = rank
        self.device = device
        self.is_ddp = is_ddp

    def log(self, msg, verbose=True, **kwargs):
        if verbose:
            print(msg, **kwargs)
        with open(os.path.join(self.log_dir, 'log.txt'), 'a') as f:
            if 'end' in kwargs:
                f.write(msg + kwargs['end'])
            else:
                f.write(msg + '\n')

    def load_ckpt(self, model, path):
        if path is not None:
            state_dict = torch.load(path, map_location=lambda storage, loc: storage)
            model_is_ddp = isinstance(model, (DDP, nn.parallel.DistributedDataParallel))
            checkpoint_has_module = any(k.startswith('module.') for k in state_dict.keys())
            if model_is_ddp and not checkpoint_has_module:
                new_state_dict = {f'module.{k}': v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict, strict=False)
            elif not model_is_ddp and checkpoint_has_module:
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict, strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)
            self.log(f'Load ckpt from: {path}')

    def save_ckpt(self, model, path):
        if self.rank == 0:
            state_dict = model.module.state_dict() if isinstance(model, (DDP, nn.parallel.DistributedDataParallel)) else model.state_dict()
            torch.save(state_dict, path)

    def set_model_train(self, model):
        model.train()

    def set_model_eval(self, model):
        model.eval()

    def train_step(self, model, loss, optimizer, batch, epoch):
        pass

    def val_step(self, model, batch):
        pass

    def test_step(self, model, batch):
        pass

    def vis_step(self, model, batch):
        pass

    def train_epoch(self, model, loss, train_dataloader, val_dataloader, optimizer, scheduler, epoch, best_criterion, best_epoch):
        if self.is_ddp:
            train_dataloader.sampler.set_epoch(epoch)
        # train
        self.set_model_train(model)
        for batch in easy_track(train_dataloader, description=f'Epoch {epoch}: Training...'):
            train_loss = self.train_step(model, loss, optimizer, batch, epoch)
        if scheduler is not None:
            if isinstance(scheduler, list):
                for s in scheduler:
                    s.step()
            else:
                scheduler.step()
        self.log(f'Epoch: {epoch}, Training loss: {train_loss:.4f} Version: {self.version}') # version: sta
        # validation
        self.set_model_eval(model)
        criterion_meter = AverageMeter()
        additional_meter = DictAvgMeter()
        for batch in easy_track(val_dataloader, description=f'Epoch {epoch}: Validating...'):
            with torch.no_grad():
                criterion, additional = self.val_step(model, batch) # 225,6, {'mse': 50912}
            criterion_meter.update(criterion, additional['n']) if 'n' in additional else criterion_meter.update(criterion)
            additional_meter.update(additional)
        current_criterion = criterion_meter.avg
        self.log(f'Epoch: {epoch}, Validation criterion: mae: {current_criterion:.4f}', end=' ')
        for k, v in additional_meter.avg.items():
            self.log(f'{k}: {v:.4f}', end=' ')
        self.log(f'Best score: {best_criterion:.4f}')
        if self.rank == 0:
            last_ckpt = glob(os.path.join(self.log_dir, 'last*.pth'))
            if last_ckpt:
                os.remove(last_ckpt[0])
        self.save_ckpt(model, os.path.join(self.log_dir, f'last.pth'))
        if current_criterion < best_criterion:
            best_criterion = current_criterion
            best_epoch = epoch
            self.log(f'Epoch {epoch}: saving best model...')
            self.save_ckpt(model, os.path.join(self.log_dir, f'best_{best_epoch}.pth'))
        return best_criterion, best_epoch
        
    def train(self, model, loss, train_dataloader, val_dataloader, optimizer, scheduler, checkpoint=None, num_epochs=100):
        self.load_ckpt(model, checkpoint)
        # model = DDP(model.to(self.device), device_ids=[self.rank]) if isinstance(model, nn.Module) else [DDP(m.to(self.device), device_ids=[self.rank]) for m in model]
        # loss
        loss = loss.to(self.device)
        best_criterion = 1e10
        best_epoch = -1
        for epoch in range(num_epochs):
            best_criterion, best_epoch = self.train_epoch(model, loss, train_dataloader, val_dataloader, optimizer, scheduler, epoch, best_criterion, best_epoch)
        self.log('Best epoch: {}, criterion: {}'.format(best_epoch, best_criterion))

    def test(self, model, test_dataloader, checkpoint=None):
        self.load_ckpt(model, checkpoint)
        # model = DDP(model.to(self.device), device_ids=[self.rank]) if isinstance(model, nn.Module) else [DDP(m.to(self.device), device_ids=[self.rank]) for m in model]
        self.set_model_eval(model)
        result_meter = DictAvgMeter()
        for batch in easy_track(test_dataloader, description='Testing...'):
            with torch.no_grad():
                result = self.test_step(model, batch)
            result_meter.update(result)
        self.log('Testing results:', end=' ')
        for key, value in result_meter.avg.items():
            if key == 'mse':
                self.log('{}: {:.4f}'.format(key, math.sqrt(value)), end=' ')
            else:
                self.log('{}: {:.4f}'.format(key, value), end=' ')
        self.log('')

    def vis(self, model, test_dataloader, checkpoint=None):
        self.load_ckpt(model, checkpoint)
        os.makedirs(os.path.join(self.log_dir, 'vis'), exist_ok=True)
        # model = DDP(model.to(self.device), device_ids=[self.rank]) if isinstance(model, nn.Module) else [DDP(m.to(self.device), device_ids=[self.rank]) for m in model]
        self.set_model_eval(model)
        for batch in easy_track(test_dataloader, description='Visualizing...'):
            with torch.no_grad():
                self.vis_step(model, batch)