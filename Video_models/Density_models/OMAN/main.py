import warnings
warnings.filterwarnings("ignore")
import argparse
import random
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import util.misc as utils
from datasets import build_dataset
from engine import train_one_epoch, evaluate
from models import build_model
import os

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
    model_without_ddp = model
    if args.distributed:
        sync_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(sync_model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    # optimizer
    param_dicts = [{"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
                   {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad], "lr": args.lr_backbone}]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epochs)
    # train and val loader
    sharing_strategy = "file_system"
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)

    def set_worker_sharing_strategy(worker_id: int) -> None:
        torch.multiprocessing.set_sharing_strategy(sharing_strategy)

    if args.type_dataset == 'SENSE':
        dataset_train = build_dataset(args.type_dataset, os.path.join(args.input_dir, 'train'), args.ann_dir, train=True)
        dataset_val = build_dataset(args.type_dataset, os.path.join(args.input_dir, 'test'), args.ann_dir, train=False)
    elif args.type_dataset == 'HT21':
        dataset_train = build_dataset(args.type_dataset, os.path.join(args.input_dir, 'train'), train=True)
        dataset_val = build_dataset(args.type_dataset, os.path.join(args.input_dir, 'val'), train=False)
    else:
        print('This dataset does not exist')
        raise NotImplementedError
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, sampler=sampler_train, num_workers=args.num_workers, pin_memory=True, worker_init_fn=set_worker_sharing_strategy)
    data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size, sampler=sampler_val, shuffle=False, num_workers=args.num_workers, pin_memory=True, worker_init_fn=set_worker_sharing_strategy)
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
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, epoch, args.clip_max_norm)
        lr_scheduler.step()
        utils.save_on_master({'model': model_without_ddp.state_dict(), 'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(), 'epoch': epoch,
                              'args': args, 'best_mae': best_mae, 'best_epoch': best_epoch}, os.path.join(args.output_dir, 'checkpoint.pth'))
        # eval
        if epoch % args.eval_freq == 0 and epoch > 1:
            results = evaluate(args, model, data_loader_val, epoch)
            mae, mse = results['mae'], results['mse']
            if mae < best_mae:
                best_epoch = epoch
                best_mae = mae
            print('Epoch: [{}/{}], MAE: {:.2f}, MSE: {:.2f}, Best MAE: {:.2f}, Best epoch: {}'.format(epoch + 1, args.epochs, mae, mse, best_mae, best_epoch + 1))
            if mae == best_mae and utils.is_main_process():
                src_path = os.path.join(args.output_dir, 'checkpoint.pth')
                dst_path = os.path.join(args.output_dir, 'best_checkpoint.pth')
                shutil.copyfile(src_path, dst_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--type_dataset', type=str, default='SENSE')
    parser.add_argument('--input_dir', type=str, default='data/Sense')
    parser.add_argument('--ann_dir', type=str, default='data/Sense/label_list_all')
    parser.add_argument('--max_len', default=3000)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--output_dir', type=str, default='saved_sense')
    # training config
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--interval', type=int, default=15)
    # model config
    parser.add_argument('--backbone', default='convnext', type=str)
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
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://')
    args = parser.parse_args()

    print('Training dataset', args.type_dataset)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    main(args)