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
    parser.add_argument('--input_dir', type=str, default='data/preprocessed_RGBT-CC')
    parser.add_argument('--output_dir', type=str, default='saved_rgbt_cc')
    # training config
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_model_num', type=int, default=1) # max model to save
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--downsample_ratio', type=int, default=8)
    parser.add_argument('--use_background', type=bool, default=True) # whether to use background modeling
    parser.add_argument('--sigma', type=float, default=8.0)
    parser.add_argument('--background_ratio', type=float, default=0.15)
    # testing config
    parser.add_argument('--val_epoch', type=int, default=1)
    parser.add_argument('--val-start', type=int, default=1)
    parser.add_argument('--save_all_best', type=bool, default=True)
    args = parser.parse_args()

    print('Training dataset:', args.type_dataset)
    trainer = RegTrainer(args)
    trainer.setup()
    trainer.train()