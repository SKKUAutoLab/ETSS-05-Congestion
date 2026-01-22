import warnings
warnings.filterwarnings("ignore")
from utils.regression_trainer import RegTrainer
import argparse
import torch
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--type_dataset', type=str, default='RGBT-CC')
    parser.add_argument('--input_dir', type=str, default='data/processed_RGBT-CC')
    parser.add_argument('--output_dir', default='saved_rgbt_cc', type=str)
    # training config
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_model_num', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--val_epoch', type=int, default=1)
    parser.add_argument('--val_start', type=int, default=20)
    parser.add_argument('--save_all_best', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--downsample_ratio', type=int, default=8)
    parser.add_argument('--use_background', type=bool, default=True)
    parser.add_argument('--sigma', type=float, default=8.0)
    parser.add_argument('--background_ratio', type=float, default=0.15)
    args = parser.parse_args()

    print('Training dataset:', args.type_dataset)
    trainer = RegTrainer(args)
    trainer.setup()
    trainer.train()