import os
from models.model_teacher_vgg import CSRNet as CSRNet_teacher
from models.model_student_vgg import CSRNet as CSRNet_student
from utils import save_checkpoint
from models.distillation import cosine_similarity, scale_process, cal_dense_fsp
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

def train(train_list, teacher, student, criterion, optimizer, epoch, args):
    losses_h = AverageMeter()
    losses_s = AverageMeter()
    losses_fsp = AverageMeter()
    losses_cos = AverageMeter()
    # train loader
    train_loader = torch.utils.data.DataLoader(dataset.listDataset(train_list, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), train=True, dataset=args.type_dataset),
                                               num_workers=args.num_workers, shuffle=True, batch_size=args.batch_size)
    teacher.eval()
    student.train()
    for i, (img, target) in enumerate(train_loader):
        img = img.cuda() # [1, 3, 664, 995]
        img = Variable(img)
        target = target.type(torch.FloatTensor).cuda() # [1, 1, 83, 125]
        target = Variable(target)
        with torch.no_grad():
            teacher_output = teacher(img) # [1, 1, 83, 125]
            teacher.features.append(teacher_output)
            teacher_fsp_features = [scale_process(teacher.features)] # [1, 64, 56, 84], [1, 64, 56, 84], [1, 128, 56, 84], [1, 256, 56, 84], [1, 512, 56, 84], [1, 256, 56, 84], [1, 1, 56, 84]
            teacher_fsp = cal_dense_fsp(teacher_fsp_features) # [64, 64], [64, 128], ..., [256, 1]
        student_features = student(img)
        student_output = student_features[-1] # [1, 1, 128, 96]
        student_fsp_features = [scale_process(student_features)]
        student_fsp = cal_dense_fsp(student_fsp_features)
        loss_h = criterion(student_output, target)
        loss_s = criterion(student_output, teacher_output)
        loss_fsp = torch.tensor([0.], dtype=torch.float).cuda()
        if args.lamb_fsp:
            loss_f = []
            assert len(teacher_fsp) == len(student_fsp)
            for t in range(len(teacher_fsp)):
                loss_f.append(criterion(student_fsp[t], teacher_fsp[t]))
            loss_fsp = sum(loss_f) * args.lamb_fsp
        loss_cos = torch.tensor([0.], dtype=torch.float).cuda()
        if args.lamb_cos:
            loss_c = []
            for t in range(len(student_features) - 1):
                loss_c.append(cosine_similarity(student_features[t], teacher.features[t]))
            loss_cos = sum(loss_c) * args.lamb_cos
        loss = loss_h + loss_s + loss_fsp + loss_cos
        losses_h.update(loss_h.item(), img.size(0))
        losses_s.update(loss_s.item(), img.size(0))
        losses_fsp.update(loss_fsp.item(), img.size(0))
        losses_cos.update(loss_cos.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % args.print_freq == (args.print_freq - 1):
            print('Epoch: [{0}][{1}/{2}], Loss h: {loss_h.avg:.4f}, Loss s: {loss_s.avg:.4f}, Loss fsp: {loss_fsp.avg:.4f}, Loss cos" {loss_kl.avg:.4f}'.format(epoch, i, len(train_loader), loss_h=losses_h, loss_s=losses_s, loss_fsp=losses_fsp, loss_kl=losses_cos))

def val(val_list, model, args):
    val_loader = torch.utils.data.DataLoader(dataset.listDataset(val_list, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), train=False, dataset=args.type_dataset),
                                             num_workers=args.num_workers, shuffle=False, batch_size=args.batch_size)
    model.eval()
    mae = 0
    mse = 0
    for i, (img, target) in enumerate(val_loader):
        img = img.cuda() # [1, 3, 688, 1024]
        img = Variable(img)
        with torch.no_grad():
            output = model(img) # [1, 1, 86, 128]
        mae += abs(output.data.sum() - target.sum().type(torch.FloatTensor).cuda())
        mse += (output.data.sum() - target.sum().type(torch.FloatTensor).cuda()).pow(2)
    N = len(val_loader)
    mae = mae / N
    mse = torch.sqrt(mse / N)
    print('Val: MAE: {:.4f}, MSE: {:.4f}'.format(mae, mse))
    return mae, mse

def test(test_list, model, args):
    test_loader = torch.utils.data.DataLoader(dataset.listDataset(test_list, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), train=False, dataset=args.type_dataset),
                                              num_workers=args.num_workers, shuffle=False, batch_size=args.batch_size)
    model.eval()
    mae = 0
    mse = 0
    for i, (img, target) in enumerate(test_loader):
        img = img.cuda() # [1, 3, 378, 810]
        img = Variable(img)
        with torch.no_grad():
            output = model(img) # [1, 1, 48, 102]
        mae += abs(output.data.sum() - target.sum().type(torch.FloatTensor).cuda())
        mse += (output.data.sum() - target.sum().type(torch.FloatTensor).cuda()).pow(2)
    N = len(test_loader)
    mae = mae / N
    mse = torch.sqrt(mse / N)
    print('Test: MAE: {:.4f}, MSE: {:.4f}'.format(mae, mse))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--type_dataset', type=str, default='sha', choices=['sha', 'shb', 'qnrf'])
    parser.add_argument('--train_json', type=str, default='A_train.json')
    parser.add_argument('--val_json', type=str, default='A_val.json')
    parser.add_argument('--test_json', type=str, default='A_test.json')
    parser.add_argument('--output_dir', type=str, default='saved_sha')
    parser.add_argument('--seed', type=int, default=time.time())
    parser.add_argument('--print_freq', type=int, default=400)
    # training config
    parser.add_argument('--lr', default=None, type=float)
    parser.add_argument('--teacher_ckpt', default=None, type=str)
    parser.add_argument('--student_ckpt', default=None, type=str)
    parser.add_argument('--lamb_fsp', type=float, default=None) # dense fsp loss
    parser.add_argument('--lamb_cos', type=float, default=None) # cos loss
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--momentum', type=float, default=0.95)
    parser.add_argument('--decay', type=float, default=5 * 1e-4)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    print('Training dataset:', args.type_dataset)
    mae_best_prec1 = 1e6
    mse_best_prec1 = 1e6
    with open(args.train_json, 'r') as outfile:
        train_list = json.load(outfile)
    with open(args.val_json, 'r') as outfile:
        val_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:
        test_list = json.load(outfile)
    torch.cuda.manual_seed(args.seed)
    # model
    teacher = CSRNet_teacher()
    student = CSRNet_student(ratio=4)
    teacher.regist_hook() # get teacher features
    teacher = teacher.cuda()
    student = student.cuda()
    # loss
    criterion = nn.MSELoss(size_average=False).cuda()
    # optimizer
    optimizer = torch.optim.Adam(student.parameters(), args.lr, weight_decay=args.decay)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # resume training
    if args.teacher_ckpt:
        if os.path.isfile(args.teacher_ckpt):
            checkpoint = torch.load(args.teacher_ckpt)
            teacher.load_state_dict(checkpoint['state_dict'])
            print('Load teacher ckpt from: {}'.format(args.teacher_ckpt))
        else:
            print('No teacher ckpt found')
    if args.student_ckpt:
        if os.path.isfile(args.student_ckpt):
            checkpoint = torch.load(args.student_ckpt)
            args.start_epoch = checkpoint['epoch']
            if 'best_prec1' in checkpoint.keys():
                mae_best_prec1 = checkpoint['best_prec1']
            else:
                mae_best_prec1 = checkpoint['mae_best_prec1']
            if 'mse_best_prec1' in checkpoint.keys():
                mse_best_prec1 = checkpoint['mse_best_prec1']
            student.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('Load student ckpt from: {}'.format(args.student_ckpt))
        else:
            print('No student ckpt found')
    for epoch in range(args.start_epoch, args.epochs):
        train(train_list, teacher, student, criterion, optimizer, epoch, args)
        mae_prec1, mse_prec1 = val(val_list, student, args)
        mae_is_best = mae_prec1 < mae_best_prec1
        mae_best_prec1 = min(mae_prec1, mae_best_prec1)
        mse_is_best = mse_prec1 < mse_best_prec1
        mse_best_prec1 = min(mse_prec1, mse_best_prec1)
        print('Best MAE: {:.4f}, MSE: {:.4f}'.format(mae_best_prec1, mse_best_prec1))
        save_checkpoint({'epoch': epoch + 1, 'arch': args.student_ckpt, 'state_dict': student.state_dict(), 'mae_best_prec1': mae_best_prec1, 'mse_best_prec1': mse_best_prec1, 'optimizer': optimizer.state_dict()}, mae_is_best, mse_is_best, args.output_dir)
        if mae_is_best or mse_is_best:
            test(test_list, student, args)