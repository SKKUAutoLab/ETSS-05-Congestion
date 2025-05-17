from utils.noisy_trainer import NoisyTrainer 
import argparse
import torch
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--input_dir', default='datasets/qnrf', type=str)
    parser.add_argument('--output_dir', type=str, default='saved_qnrf')
    parser.add_argument('--skip_test', default=False)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--is_gray', type=bool, default=False)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--downsample_ratio', type=int, default=8)
    parser.add_argument('--use_background', type=bool, default=True)
    parser.add_argument('--sigma', type=float, default=8.0)
    parser.add_argument('--alpha', type=float, default=8.0)
    parser.add_argument('--ratio', type=float, default=0.6)
    parser.add_argument('--background_ratio', type=float, default=0.15)
    # model config
    parser.add_argument('--net', default='vgg19', type=str)
    # loss config
    parser.add_argument('--loss', default='full', type=str)
    parser.add_argument('--add', default=True)
    parser.add_argument('--weight', type=float, default=0.01)
    parser.add_argument('--reg', default=False)
    parser.add_argument('--bn', default=False)
    parser.add_argument('--minx', type=float, default=1e-14)
    parser.add_argument('--o_cn', type=int, default=1)
    # training config
    parser.add_argument('--lr', type=float, default=3e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_model_num', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--val_epoch', type=int, default=5)
    parser.add_argument('--val_start', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    print('Training dataset:', args.input_dir.split('/')[-1])
    torch.backends.cudnn.benchmark = True
    trainer = NoisyTrainer(args)
    trainer.setup()
    trainer.train()