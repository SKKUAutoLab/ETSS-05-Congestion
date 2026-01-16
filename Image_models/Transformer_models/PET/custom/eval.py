import warnings
warnings.filterwarnings("ignore")
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from datasets import build_dataset
import util.misc as utils
from engine import evaluate
from models import build_model

def del_(cheakpoint):
    del cheakpoint['backbone.0.backbone.layers.0.blocks.1.attn_mask']
    del cheakpoint['backbone.0.backbone.layers.1.blocks.1.attn_mask']
    del cheakpoint['backbone.0.backbone.layers.3.blocks.0.attn.relative_coords_table']
    del cheakpoint['backbone.0.backbone.layers.3.blocks.0.attn.relative_position_index']
    del cheakpoint['backbone.0.backbone.layers.3.blocks.1.attn.relative_coords_table']
    del cheakpoint['backbone.0.backbone.layers.3.blocks.1.attn.relative_position_index']
    return cheakpoint

def main(args):
    utils.init_distributed_mode(args)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # model
    model, criterion = build_model(args)
    model.cuda()
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    # val loader
    dataset_val = build_dataset(image_set='val', args=args)
    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    # load pretrained model
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
            if args.backbone=="swin_v2":
               checkpoint['model'] = del_(checkpoint['model'])
        model_without_ddp.load_state_dict(checkpoint['model'],False)
        cur_epoch = checkpoint['epoch']
        print('Load ckpt from:', args.resume)
    vis_dir = None if args.vis_dir == "" else args.vis_dir
    test_stats = evaluate(model, data_loader_val, criterion=criterion, vis_dir=vis_dir, args=args)
    mae, mse, P, R, F, abs = test_stats['mae'], test_stats['mse'], test_stats['Precision_s'], test_stats['Recall_s'], test_stats['F1_s'], test_stats['abs']
    print('Epoch: {}, MAE: {:.2f}, MSE: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}, Abs: {:.2f}'.format(cur_epoch, mae, mse, P, R, F, abs))

def test_main(args, resume):
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # model
    model, criterion = build_model(args)
    model.cuda()
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    # val loader
    dataset_val = build_dataset(image_set='val', args=args)
    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    # load pretrained model
    if resume:
        checkpoint = torch.load(resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'],False)
        print('Load ckpt from:', resume)
    test_stats = evaluate(model, data_loader_val, criterion=criterion, args=args)
    mae, mse = test_stats['mae'], test_stats['mse']
    return mae, mse

def test(args, resume):
    mae, mse = test_main(args, resume)
    return mae, mse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--type_dataset', type=str, default='sha')
    parser.add_argument('--input_dir', type=str, default='data/ShanghaiTech/part_A')
    parser.add_argument('--pretrained', default="pretrained/vgg16_bn-6c64b313.pth", type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='saved_sha/best_checkpoint.pth', type=str)
    parser.add_argument('--vis_dir', default="vis_sha")
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://')
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
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    main(args)