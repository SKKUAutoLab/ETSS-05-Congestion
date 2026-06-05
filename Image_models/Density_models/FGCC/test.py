import warnings
warnings.filterwarnings('ignore')
from utils.utils import Validator
from models.model import Model
from datasets.data_loader import DataLoader
import argparse

def main(args):
    # test loader
    data_loader = DataLoader(args)
    # model
    model = Model(args).get_model()
    if args.ckpt_dir != '':
        model.load(args.ckpt_dir)
    if args.ckpt_dir_counter != '':
        model.load_counter(args.ckpt_dir_counter)
    tester = Validator(suffix='test')
    tester.best_p = 0
    tester.validate(model, data_loader.get_test_loader())
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--type_dataset', type=str, default='Towards_vs_Away')
    parser.add_argument('--input_dir', type=str, default='datasets/Towards_vs_Away')
    parser.add_argument('--downsample', type=int, default=0)
    parser.add_argument('--roi', default=False, action='store_true')
    parser.add_argument('--gray', default=False, action='store_true')
    parser.add_argument('--smap', default=False, action='store_true')
    parser.add_argument('--dmap_type', type=str, default='dot')
    parser.add_argument('--seg_gt_act', type=str, default='ignore')
    parser.add_argument('--loader', type=str, default='single')
    parser.add_argument('--ckpt_dir', type=str, default='') # load checkpoint
    parser.add_argument('--ckpt_dir_counter', type=str, default='') # load counter checkpoint
    parser.add_argument('--nThreads', default=1, type=int)
    parser.add_argument('--output_dir', type=str, default='saved_towards_vs_away')
    parser.add_argument('--display_freq', type=int, default=100)
    # model config
    parser.add_argument('--model', type=str, default='direct_regression')
    parser.add_argument('--net', type=str, default='fcn')
    parser.add_argument('--input_cn', type=int, default=3)
    parser.add_argument('--output_cn', type=int, default=2)
    parser.add_argument('--hard_assign', type=bool, default=False)
    parser.add_argument('--ignore_index', type=int, default=-1)
    parser.add_argument('--hourglass_iter', type=int, default=1)
    parser.add_argument('--multi_reg', default=False, action='store_true')
    # training config
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--seg_lr', type=float, default=0.0001)
    parser.add_argument('--weight', type=float, default=1)
    parser.add_argument('--seg_w', type=float, default=1)
    parser.add_argument('--scale', type=int, default=16)
    parser.add_argument('--grid', type=int, default=7)
    parser.add_argument('--align', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--train_shuffle', type=bool, default=True)
    parser.add_argument('--train_counter', default=False, action='store_true')
    parser.add_argument('--seg_loss', default=False, action='store_true')
    parser.add_argument('--final_loss', default=False, action='store_true')
    parser.add_argument('--count_loss', default=False, action='store_true')
    parser.add_argument('--prop', default=False, action='store_true')
    parser.add_argument('--seg', default=False, action='store_true')
    parser.add_argument('--soft', default=False, action='store_true')
    parser.add_argument('--per', default=False, action='store_true')
    parser.add_argument('--att', default=False, action='store_true')
    # testing config
    parser.add_argument('--val_freq', type=int, default=1)
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--test_freq', type=int, default=5)
    parser.add_argument('--save_freq', type=int, default=5)
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    main(args)