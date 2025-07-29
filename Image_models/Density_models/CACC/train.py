from model import CANNet
from utils import save_checkpoint
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import argparse
import json
import dataset
import time
import warnings
warnings.filterwarnings("ignore")

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

def train(train_list, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_loader = torch.utils.data.DataLoader(dataset.listDataset(train_list, shuffle=True, transform=transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), train=True, seen=model.seen,
                                               batch_size=args.batch_size, num_workers=args.num_workers), batch_size=args.batch_size)
    model.train()
    end = time.time()
    for i,(img, target)in enumerate(train_loader):
        data_time.update(time.time() - end)
        img = img.cuda() # [26, 3, 384, 512]
        img = Variable(img)
        output = model(img)[:, 0, :, :] # [26, 48, 64]
        target = target.type(torch.FloatTensor).cuda() # [26, 48, 64]
        target = Variable(target)
        loss = criterion(output, target)
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}], Time: {batch_time.val:.3f} ({batch_time.avg:.3f}), Data {data_time.val:.3f} ({data_time.avg:.3f}), Loss {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses))

def validate(val_list, model):
    val_loader = torch.utils.data.DataLoader(dataset.listDataset(val_list, shuffle=False, transform=transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),  train=False), batch_size=1)
    model.eval()
    mae = 0
    for i, (img, target) in enumerate(val_loader):
        h, w = img.shape[2:4]
        h_d = h // 2
        w_d = w // 2
        img_1 = Variable(img[:,:,:h_d,:w_d].cuda())
        img_2 = Variable(img[:,:,:h_d,w_d:].cuda())
        img_3 = Variable(img[:,:,h_d:,:w_d].cuda())
        img_4 = Variable(img[:,:,h_d:,w_d:].cuda())
        density_1 = model(img_1).data.cpu().numpy()
        density_2 = model(img_2).data.cpu().numpy()
        density_3 = model(img_3).data.cpu().numpy()
        density_4 = model(img_4).data.cpu().numpy()
        pred_sum = density_1.sum()+density_2.sum()+density_3.sum()+density_4.sum()
        mae += abs(pred_sum-target.sum())
    mae = mae / len(val_loader)
    print('MAE: {:.4f}'.format(mae))
    return mae

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_json', type=str, default='datasets/train_part_A.json')
    parser.add_argument('--val_json', type=str, default='datasets/test_part_A.json')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=26)
    parser.add_argument('--decay', type=float, default=5 * 1e-4)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=int(time.time()))
    parser.add_argument('--print_freq', type=int, default=4)
    args = parser.parse_args()

    best_prec1 = 1e6
    with open(args.train_json, 'r') as outfile:
        train_list = json.load(outfile)
    with open(args.val_json, 'r') as outfile:
        val_list = json.load(outfile)
    torch.cuda.manual_seed(args.seed)
    # model
    model = CANNet()
    model = model.cuda()
    # loss
    criterion = nn.MSELoss(size_average=False).cuda()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.decay)
    for epoch in range(args.start_epoch, args.epochs):
        train(train_list, model, criterion, optimizer, epoch)
        prec1 = validate(val_list, model)
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print('Best MAE: {:.4f}'.format(best_prec1))
        save_checkpoint({'state_dict': model.state_dict()}, is_best)