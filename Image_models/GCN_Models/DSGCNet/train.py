import warnings
warnings.filterwarnings('ignore')
import argparse
import random
from torch.utils.data import DataLoader
from torch import nn
from crowd_datasets import build_dataset
from engine import train_one_epoch, evaluate_crowd_no_overlap
import numpy as np
import util.misc as utils
import torch
from models import build_model
import os
torch.multiprocessing.set_sharing_strategy('file_system') # or set num_workers=0 to avoid training error

def setup_seed(args):
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(args):
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    setup_seed(args)
    # model
    model, criterion = build_model(args, training=True)
    model.cuda()
    criterion.cuda()
    # loss
    density_criterion = nn.MSELoss(size_average=False).cuda()
    # optimizer
    model_without_ddp = model
    param_dicts = [{"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
                   {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad], "lr": args.lr_backbone}]
    optimizer = torch.optim.Adam(param_dicts, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    # train and test loader
    loading_data = build_dataset(args=args)
    train_set, val_set = loading_data(args.input_dir, args.type_dataset)
    sampler_train = torch.utils.data.RandomSampler(train_set)
    sampler_val = torch.utils.data.SequentialSampler(val_set)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(train_set, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn_crowd_train, num_workers=args.num_workers)
    data_loader_val = DataLoader(val_set, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)
    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])
    # resume training
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
    mae = []
    mse = []
    density_mae = []
    density_mse = []
    for epoch in range(args.start_epoch, args.epochs):
        stat = train_one_epoch(model, criterion, data_loader_train, optimizer, density_criterion, args.clip_max_norm)
        print('Epoch: [{}/{}], Loss sum: {:.4f}, Loss: {:.4f}, Den loss: {:.4f}, CE loss: {:.4f}'.format(epoch + 1, args.epochs, stat['loss_sum'], stat['losses'], stat['den_loss'], stat['loss_ce']))
        lr_scheduler.step()
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        checkpoint_latest_path = os.path.join(args.output_dir, f'latest.pth')
        torch.save({'model': model_without_ddp.state_dict()}, checkpoint_latest_path)
        if epoch % args.eval_freq == 0 and epoch != 0:
            result = evaluate_crowd_no_overlap(model, data_loader_val)
            mae.append(result[0])
            mse.append(result[1])
            density_mae.append(result[2])
            density_mse.append(result[3])
            print('Epoch: [{}/{}], MAE: {:.2f}, MSE: {:.2f}, Best MAE: {:.2f}, Density MAE: {:.2f}, Density MSE: {:.2f}, Best density MAE: {:.2f}'.format(epoch + 1, args.epochs, result[0], result[1], np.min(mae), result[2], result[3], np.min(density_mae)))
            if abs(np.min(mae) - result[0]) < 0.01:
                checkpoint_best_path = os.path.join(args.output_dir, f'best_{epoch + 1}.pth')
                torch.save({'model': model_without_ddp.state_dict(),}, checkpoint_best_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--type_dataset', type=str, default='sha', choices=['sha', 'shb'])
    parser.add_argument('--input_dir', type=str, default='datasets/ShanghaiTech/part_A')
    parser.add_argument('--output_dir', type=str, default='saved_sha')
    parser.add_argument('--seed', default=42, type=int)
    # training config
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=2500, type=int)
    parser.add_argument('--lr_drop', default=800, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)
    parser.add_argument('--resume', default='')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    # model config
    parser.add_argument('--frozen_weights', type=str, default=None)
    parser.add_argument('--backbone', default='vgg16_bn', type=str)
    parser.add_argument('--set_cost_class', default=1, type=float)
    parser.add_argument('--set_cost_point', default=0.05, type=float)
    parser.add_argument('--point_loss_coef', default=0.0002, type=float)
    parser.add_argument('--eos_coef', default=0.5, type=float)
    parser.add_argument('--row', default=2, type=int)
    parser.add_argument('--line', default=2, type=int)
    # testing config
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_freq', default=1, type=int)
    args = parser.parse_args()

    print('Training dataset:', args.type_dataset)
    main(args)