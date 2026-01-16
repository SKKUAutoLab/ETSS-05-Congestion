import warnings
warnings.filterwarnings("ignore")
import argparse
import random
import os
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import util.misc as utils
from datasets import build_dataset
from engine import evaluate, train_one_epoch
from models import build_model

def setup_seed(args):
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(args):
    utils.init_distributed_mode(args)
    setup_seed(args)
    # model
    model, criterion = build_model(args)
    model.cuda()
    if args.syn_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters:', n_parameters / 1e6)
    # optimizer
    param_dicts = [{"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
                   {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad], "lr": args.lr_backbone}]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epochs)
    # train and test loader
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    # resume training
    best_mae, best_epoch = 1e8, 0
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            best_mae = checkpoint['best_mae']
            best_epoch = checkpoint['best_epoch']
        print('Load ckpt from:', args.resume)
    # train
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, epoch, args.clip_max_norm)
        lr_scheduler.step()
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        checkpoint_path = os.path.join(args.output_dir, 'checkpoint.pth')
        utils.save_on_master({'model': model_without_ddp.state_dict(), 'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(), 'epoch': epoch,
                              'args': args, 'best_mae': best_mae}, checkpoint_path)
        # test
        if epoch % args.eval_freq == 0 and epoch > 0:
            test_stats = evaluate(model, data_loader_val, args.vis_dir)
            mae, mse = test_stats['mae'], test_stats['mse']
            if mae < best_mae:
                best_epoch = epoch
                best_mae = mae
            print('Epoch: [{}/{}], MAE: {:.2f}, MSE: {:.2f}, Best MAE: {:.2f}, Best epoch: {}'.format(epoch + 1, args.epochs, mae, mse, best_mae, best_epoch))
            if mae == best_mae and utils.is_main_process():
                src_path = os.path.join(args.output_dir, 'checkpoint.pth')
                dst_path = os.path.join(args.output_dir, 'best_checkpoint.pth')
                shutil.copyfile(src_path, dst_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--type_dataset', type=str, default='sha')
    parser.add_argument('--input_dir', type=str, default='data/ShanghaiTech/part_A')
    parser.add_argument('--output_dir', type=str, default='saved_sha')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--vis_dir', type=str, default='vis_sha')
    # training config
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=1500, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    # model config
    parser.add_argument('--backbone', default='vgg16_bn', type=str)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned', 'fourier'))
    parser.add_argument('--dec_layers', default=2, type=int)
    parser.add_argument('--dim_feedforward', default=512, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    # loss config
    parser.add_argument('--set_cost_class', default=1, type=float)
    parser.add_argument('--set_cost_point', default=0.05, type=float)
    parser.add_argument('--ce_loss_coef', default=1.0, type=float)
    parser.add_argument('--point_loss_coef', default=5.0, type=float)
    parser.add_argument('--eos_coef', default=0.5, type=float)
    # testing config
    parser.add_argument('--eval_freq', default=5, type=int)
    parser.add_argument('--syn_bn', default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://')
    args = parser.parse_args()

    print('Training dataset:', args.type_dataset)
    main(args)