import argparse
import os
import torch
from train_helper_JCTNet import Trainer


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--data-dir', default='/home/hdd/dataset_xzw/NWPU-Crowd-384x1920', help='data path')
    parser.add_argument('--dataset', default='nwpu', help='dataset name: qnrf, nwpu, sha, shb')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='the initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='the weight decay')
    parser.add_argument('--resume', default='', type=str,
                        help='the path of resume training model')
    parser.add_argument('--max-epoch', type=int, default=4000,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=100,
                        help='the epoch start to val')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='train batch size')
    parser.add_argument('--device', default='3', help='assign device')
    parser.add_argument('--num-workers', type=int, default=16,
                        help='the num of training process')
    parser.add_argument('--crop-size', type=int, default= 256,
                        help='the crop size of the train image')

    args = parser.parse_args()

    if args.dataset.lower() == 'qnrf':
        args.crop_size = 512
    elif args.dataset.lower() == 'nwpu':
        args.crop_size = 256
        args.val_epoch = 10
    elif args.dataset.lower() == 'sha':
        args.crop_size = 256
    elif args.dataset.lower() == 'shb':
        args.crop_size = 512
    else:
        raise NotImplementedError
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = Trainer(args)
    trainer.setup()
    trainer.train()

