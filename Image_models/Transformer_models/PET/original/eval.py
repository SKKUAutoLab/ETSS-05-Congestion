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
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters:', n_parameters / 1e6)
    # test loader
    dataset_val = build_dataset(image_set='val', args=args)
    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    # load trained model
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])        
        cur_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
        print('Load ckpt from:', args.resume)
    # test
    vis_dir = None if args.vis_dir == "" else args.vis_dir
    test_stats = evaluate(model, data_loader_val, vis_dir=vis_dir)
    mae, mse = test_stats['mae'], test_stats['mse']
    print('Epoch: {}, MAE: {:.2f}, MSE: {:.2f}'.format(cur_epoch, mae, mse))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--type_dataset', type=str, default='sha')
    parser.add_argument('--input_dir', type=str, default='data/ShanghaiTech/part_A')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='saved_sha/best_checkpoint.pth', type=str)
    parser.add_argument('--vis_dir', default="vis_sha", type=str)
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
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://')
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    main(args)