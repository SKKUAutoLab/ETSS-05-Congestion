from utils.emd_dot_trainer import EMDTrainer 
import argparse
import torch
import warnings
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--input_dir', default='datasets/qnrf', type=str)
    parser.add_argument('--output_dir', default='saved_qnrf', type=str)
    parser.add_argument('--resume', default='', type=str)
    # model config
    parser.add_argument('--o_cn', type=int, default=1)
    parser.add_argument('--cost', type=str, default='per')
    parser.add_argument('--scale', type=float, default=0.6)
    parser.add_argument('--reach', type=float, default=0.5)
    parser.add_argument('--blur', type=float, default=0.01)
    parser.add_argument('--scaling', type=float, default=0.5)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--p', type=float, default=1)
    parser.add_argument('--d_point', type=str, default='l1')
    parser.add_argument('--d_pixel', type=str, default='l2')
    # training config
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_model_num', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--val_epoch', type=int, default=5)
    parser.add_argument('--val_start', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--is_gray', type=bool, default=False)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--downsample_ratio', type=int, default=8)
    args = parser.parse_args()

    print('Training dataset:', args.input_dir.split('/')[-1])
    trainer = EMDTrainer(args)
    trainer.setup()
    trainer.train()