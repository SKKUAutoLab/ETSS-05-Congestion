import torch.nn as nn
import torch
import os
import numpy as np
from torch.autograd import Variable
from collections import OrderedDict
from models.networks import HOUR_GLASS_PROP, CSRNet_TWO

class SegAttPropVGG():
    def __init__(self, args):
        self.output_dir = args.output_dir
        self.args = args
        att_cn = 1 if args.att else 0
        # model
        if args.net == 'vgg':
            self.init_predictor = CSRNet_TWO(args.input_cn, args.output_cn + 1).cuda()
            self.seg_refiner = HOUR_GLASS_PROP(64, args.output_cn + 1, _iter=args.hourglass_iter, att_cn=att_cn, prop=args.prop).cuda()
            self.cnt_refiner = HOUR_GLASS_PROP(64, 1, _iter=1, att_cn=att_cn).cuda()
        else:
            print('This model does not exist')
            raise NotImplementedError
        self.softmax = nn.Softmax2d()
        # loss
        self.mse = nn.MSELoss(reduction='sum').cuda()
        # optimizer
        params = list(self.init_predictor.parameters()) 
        self.optimizer = torch.optim.Adam(params, args.lr)
        weight = torch.Tensor([0.3158, 0.6684, 0.0156])
        self.weight = args.weight
        self.cls_weight = weight.reshape(1, 3, 1, 1).cuda()
        params = list(self.seg_refiner.parameters()) + list(self.cnt_refiner.parameters())
        self.seg_optimizer = torch.optim.Adam(params, args.seg_lr)
        self.display_num = 0
        self.display_count_loss = 0
        self.display_seg_loss = 0
        self.display_acc = 0
        self.display_global_err = 0
        self.display_final_loss = 0

    def reset(self):
        self.display_num = 0
        self.display_count_loss = 0
        self.display_seg_loss = 0
        self.display_acc = 0
        self.display_global_err = 0
        self.display_final_loss = 0

    def set_data(self, data):
        self.img = Variable(data['img'].cuda()) # [1, 3, 256, 256]
        self.target = Variable(data['den'].type(torch.FloatTensor).cuda()) # [1, 2, 256, 256]
        s = self.target.sum()
        if self.args.downsample:
            size = [int(self.img.shape[2] / self.args.downsample), int(self.img.shape[3] / self.args.downsample)] # [32, 32]
            self.target = nn.functional.interpolate(self.target, size) # [1, 2, 32, 32]
            self.target = self.target / self.target.sum() * s if s > 0 else self.target # [1, 2, 32, 32]
        self.smap_target = self.target / (torch.sum(self.target, dim=1).unsqueeze(1) + 1e-14) # [1, 2, 32, 32]
        self.smap_target[:, 0, :, :][torch.sum(self.target, dim=1) < 0.00001] = 0
        self.smap_target[:, 1, :, :][torch.sum(self.target, dim=1) < 0.00001] = 0
        background = (torch.sum(self.smap_target, dim=1) < 0.00001).type(torch.cuda.FloatTensor) # [1, 32, 32]
        self.smap_target = torch.cat((self.smap_target, background.unsqueeze(1)), 1).type(torch.cuda.FloatTensor) # [1, 3, 32, 32]
        if self.args.roi and self.args.downsample:
            self.mask = data['mask'].type(torch.FloatTensor).cuda() # [1, 1, 256, 256]
            self.mask = nn.functional.interpolate(self.mask, size=size) # [1, 1, 32, 32]
            self.target = torch.mul(self.mask, self.target) # [1, 2, 32, 32]
            self.smap_target = torch.mul(self.mask, self.smap_target) # [1, 3, 32, 32]
        self.img.requires_grad = True

    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        self.seg_optimizer.zero_grad()
        self.backward()
        if self.args.train_counter:
            self.optimizer.step()
        self.seg_optimizer.step()
        self.display_num += 1
        self.display_count_loss += self.count_loss.detach()
        self.display_seg_loss += self.seg_loss.detach()
        self.display_global_err += abs(self.dmap.detach().sum() - self.target.detach().sum())
        self.display_acc += self.get_acc_prec_recall(self.smap.detach(), self.smap_target)[0]
        self.display_final_loss += self.final_loss.detach()
    
    def forward(self):
        self.init_dmap, self.init_smap = self.init_predictor(self.img) # [1, 1, 32, 32], [1, 3, 32, 32]
        _dmap = self.init_dmap.detach()
        self.smap = self.seg_refiner(self.init_predictor.smap_fea, _dmap)
        _smap = self.softmax(self.smap.detach())[:, 0:-1, :, :].sum(1).unsqueeze(1)
        self.dmap = self.cnt_refiner(self.init_predictor.dmap_fea, _smap)
        self.seg_output = self.smap
        self.output = torch.mul(self.dmap, self.softmax(self.smap[:, 0:-1, :, :])) # [1, 2, 32, 32]
        if self.args.roi:
            if self.args.downsample:
                size = [int(self.dmap.shape[2]), int(self.dmap.shape[3])]
                self.mask = nn.functional.interpolate(self.mask, size)
            self.output = torch.mul(self.mask, self.output) # [1, 2, 32, 32]
            self.init_smap = torch.mul(self.mask, self.init_smap)
            self.init_dmap = torch.mul(self.mask, self.init_dmap)
            self.dmap = torch.mul(self.mask, self.dmap)
            self.smap = torch.mul(self.mask, self.smap)
        return self.output

    def soft_corss_entropy(self, output, target, cls_weight=None, pixel_weight=None): # [1, 3, 32, 32], [1, 3, 32, 32], [1, 3, 1, 1], [1, 32, 32]
        output = -nn.functional.log_softmax(output, dim=1)
        if cls_weight is not None:
            output = output * cls_weight.reshape(-1, 1, 1)
            target = target * cls_weight.reshape(-1, 1, 1)
        if pixel_weight is not None:
            output = output * pixel_weight
            target = target * pixel_weight
        loss = torch.mean(torch.mul(output, target))
        return loss

    def backward(self):
        w = 1 if self.args.count_loss else 0
        self.count_loss = w * self.mse(self.dmap, torch.sum(self.target, dim=1).unsqueeze(1))
        self.count_loss += w * self.mse(self.init_dmap, torch.sum(self.target, dim=1).unsqueeze(1))
        pixel_weight = torch.sum(self.target, dim=1) / torch.sum(self.target, dim=1).max() if self.args.seg else 1
        w = self.args.seg_w if self.args.seg_loss else 0
        self.seg_loss = w * self.soft_corss_entropy(self.smap, self.smap_target, self.cls_weight, pixel_weight)
        self.seg_loss += w * self.soft_corss_entropy(self.init_smap, self.smap_target, self.cls_weight, pixel_weight)
        self.weight = self.weight if self.args.final_loss else 0
        self.final_loss = self.weight * self.mse(self.output, self.target)
        self.loss = self.count_loss + self.seg_loss + self.final_loss
        self.loss.backward()

    def get_acc_prec_recall(self, output, target): # [1, 3, 32, 32], [1, 3, 32, 32]
        acc = 0
        prec = []
        recall = []
        _, smap = torch.max(output, dim=1)
        _, target = torch.max(target, dim=1)
        a = 0.0
        b = 0.0
        for i in range(0, self.args.output_cn):
            tp = torch.sum(smap[target==i]==i)
            a += tp
            b += torch.sum(target==i)
            prec.append(tp.type(torch.FloatTensor) / (torch.sum((smap==i)).type(torch.FloatTensor) + 1))
            recall.append(tp.type(torch.FloatTensor) / (torch.sum(target==i).type(torch.FloatTensor) + 1))
        b += 1
        acc += a.type(torch.FloatTensor) / b.type(torch.FloatTensor) * 100
        return acc, np.array(prec), np.array(recall)
    
    def validate(self, data):
        self.set_data(data)
        output = self.forward()
        results = OrderedDict()
        p = 0
        relative_mae = []
        for i in range(self.args.output_cn):
            temp = self.target[:, i, :, :].sum().cpu().numpy() # [1]
            res = abs(output[:, i, :, :].sum() - self.target[:, i, :, :].sum()).cpu().data.numpy()
            results['Group_%i' % i] = res
            p += res
            if temp > 0:
                relative_mae.append(res / temp)
            else:
                relative_mae.append(0)
        return p, results, np.array(relative_mae)

    def get_info(self):
        info = OrderedDict()
        info['count_loss'] = self.display_count_loss / self.display_num
        info['seg_loss'] = self.display_seg_loss / self.display_num
        info['seg_acc'] = self.display_acc / self.display_num
        info['global_error'] = self.display_global_err / self.display_num
        info['final_loss'] = self.display_final_loss / self.display_num
        return info

    def display(self):
        info = self.get_info()
        for k in info.keys():
            print("%s: %.3f, " % (k, info[k]), end='')

    def save(self, suffix):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        path = os.path.join(self.output_dir, suffix + '.pth')
        torch.save({'state_dict_init_predictor': self.init_predictor.state_dict(), 'state_dict_seg_refiner': self.seg_refiner.state_dict(),
                    'state_dict_cnt_refiner': self.cnt_refiner.state_dict()}, path)
    
    def load_counter(self, suffix):
        if '.pth' in suffix:
            path = suffix.rstrip()
        else:
            path = os.path.join(self.output_dir, suffix + '.pth')
        checkpoint = torch.load(path)
        self.init_predictor.load_state_dict(checkpoint['state_dict_init_predictor'])
        print('Load ckpt from:', path)

    def load(self, suffix):
        if '.pth' in suffix:
            path = suffix.rstrip()
        else:
            path = os.path.join(self.output_dir, suffix + '.pth')
        checkpoint = torch.load(path.strip())
        self.init_predictor.load_state_dict(checkpoint['state_dict_init_predictor'])
        self.seg_refiner.load_state_dict(checkpoint['state_dict_seg_refiner'])
        self.cnt_refiner.load_state_dict(checkpoint['state_dict_cnt_refiner'])
        print('Load ckpt from:', path)