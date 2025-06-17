from Networks.models import base_patch16_384_token, base_patch16_384_gap
import torch.nn as nn
from torchvision import transforms
import dataset
import math
from utils import save_checkpoint, setup_seed
import torch
import os
import nni
from nni.utils import merge_parameter
from config import return_args, args
import numpy as np
from image import load_data
import time
import warnings
warnings.filterwarnings('ignore')

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def pre_data(train_list):
    data_keys = {}
    count = 0
    for j in range(len(train_list)):
        Img_path = train_list[j]
        fname = os.path.basename(Img_path)
        img, gt_count = load_data(Img_path)
        blob = {}
        blob['img'] = img
        blob['gt_count'] = gt_count
        blob['fname'] = fname
        data_keys[count] = blob
        count += 1
    return data_keys

def main(args):
    # train and test list
    if args['type_dataset'] == 'sha':
        train_file = 'npydata/ShanghaiA_train.npy'
        test_file = 'npydata/ShanghaiA_test.npy'
    elif args['type_dataset'] == 'shb':
        train_file = 'npydata/ShanghaiB_train.npy'
        test_file = 'npydata/ShanghaiB_test.npy'
    elif args['type_dataset'] == 'qnrf':
        train_file = 'npydata/qnrf_train.npy'
        test_file = 'npydata/qnrf_test.npy'
    else:
        print('This dataset does not exist')
        raise NotImplementedError
    with open(train_file, 'rb') as outfile:
        train_list = np.load(outfile).tolist()
    with open(test_file, 'rb') as outfile:
        val_list = np.load(outfile).tolist()
    # model
    if args['type_model'] == 'token':
        model = base_patch16_384_token(pretrained=True)
    elif args['type_model'] == 'gap':
        model = base_patch16_384_gap(pretrained=True)
    else:
        print('This model does not exist')
        raise NotImplementedError
    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()
    # loss
    criterion = nn.L1Loss(size_average=False).cuda()
    # optimizer
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': args['lr']}], lr=args['lr'], weight_decay=args['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.1, last_epoch=-1)
    if not os.path.exists(args['output_dir']):
        os.makedirs(args['output_dir'])
    if args['pre']:
        if os.path.isfile(args['pre']):
            checkpoint = torch.load(args['pre'])
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_prec1']
            print('Load ckpt from: {}'.format(args['pre']))
        else:
            print("No ckpt found at {}".format(args['pre']))
    torch.set_num_threads(args['num_workers'])
    train_data = pre_data(train_list)
    test_data = pre_data(val_list)
    for epoch in range(args['start_epoch'], args['epochs']):
        train(train_data, model, criterion, optimizer, epoch, args, scheduler)
        if epoch % 5 == 0 and epoch >= 10:
            prec1 = validate(test_data, model, args)
            is_best = prec1 < args['best_pred']
            args['best_pred'] = min(prec1, args['best_pred'])
            print('Best MAE: {:.4f}'.format(args['best_pred']))
            save_checkpoint({'epoch': epoch + 1, 'arch': args['pre'], 'state_dict': model.state_dict(), 'best_prec1': args['best_pred'], 'optimizer': optimizer.state_dict()}, is_best, args['output_dir'])

def train(Pre_data, model, criterion, optimizer, epoch, args, scheduler):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # train loader
    train_loader = torch.utils.data.DataLoader(dataset.listDataset(Pre_data, args['output_dir'], transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), train=True), batch_size=args['batch_size'], drop_last=False)
    args['lr'] = optimizer.param_groups[0]['lr']
    print('Epoch: %d, Processed: %d samples, lr: %.10f' % (epoch, epoch * len(train_loader.dataset), args['lr']))
    model.train()
    end = time.time()
    for i, (fname, img, gt_count) in enumerate(train_loader):
        data_time.update(time.time() - end)
        img = img.cuda() # [8, 3, 384, 384]
        out1 = model(img) # [8, 1]
        gt_count = gt_count.type(torch.FloatTensor).cuda().unsqueeze(1) # [8, 1]
        loss = criterion(out1, gt_count)
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args['print_freq'] == 0:
            print('Epoch: [{0}][{1}/{2}], Time: {batch_time.val:.4f} ({batch_time.avg:.4f}), Data: {data_time.val:.4f} ({data_time.avg:.4f}), Loss: {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses))
    scheduler.step()

def validate(Pre_data, model, args):
    batch_size = 1
    # test loader
    test_loader = torch.utils.data.DataLoader(dataset.listDataset(Pre_data, args['output_dir'], transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), train=False), batch_size=batch_size)
    model.eval()
    mae = 0.0
    mse = 0.0
    for i, (fname, img, gt_count) in enumerate(test_loader):
        img = img.cuda() # [1, 6, 3, 384, 384]
        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        with torch.no_grad():
            out1 = model(img) # [6, 1]
            count = torch.sum(out1).item()
        gt_count = torch.sum(gt_count).item()
        mae += abs(gt_count - count)
        mse += abs(gt_count - count) * abs(gt_count - count)
        if i % 15 == 0:
            print('{fname}, GT: {gt:.4f}, Pred: {pred}'.format(fname=fname[0], gt=gt_count, pred=count))
    mae = mae * 1.0 / (len(test_loader) * batch_size)
    mse = math.sqrt(mse / (len(test_loader)) * batch_size)
    nni.report_intermediate_result(mae)
    print('MAE: {:.4f}, MSE: {:.4f}'.format(mae, mse))
    return mae

if __name__ == '__main__':
    setup_seed(args.seed)
    tuner_params = nni.get_next_parameter()
    params = vars(merge_parameter(return_args, tuner_params))
    main(params)