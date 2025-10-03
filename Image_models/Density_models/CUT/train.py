import argparse
import torch
from utils.regression_trainer import Reg_Trainer
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--type_dataset', type=str, default='sha')
    parser.add_argument('--seed', default=15, type=int)
    parser.add_argument('--crop_size', default=512, type=int)
    parser.add_argument('--input_dir', default='processed_datasets/sha', type=str)
    parser.add_argument('--output_dir', default='saved_sha', type=str)
    parser.add_argument('--max_num', default=1, type=int) # max num for saving model
    parser.add_argument('--resume', default="", type=str)
    # model config
    parser.add_argument('--downsample_ratio', default=8, type=int)
    parser.add_argument('--pretrained', default='weights/pcpvt_large.pth', type=str)
    parser.add_argument('--drop', type=float, default=0.0)
    parser.add_argument('--drop_path', type=float, default=0.45)
    # training config
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--gamma', default=2, type=float)  # focal loss
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--opt_eps', default=1e-8, type=float)
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    # testing config
    parser.add_argument('--start_val', default=200, type=int)
    parser.add_argument('--val_epoch', default=1, type=int)
    args = parser.parse_args()

    print('Training dataset:', args.type_dataset)
    trainer = Reg_Trainer(args)
    trainer.setup()
    trainer.train()