import argparse

parser = argparse.ArgumentParser()
# general config
parser.add_argument('--type_dataset', type=str, default='sha', choices=['sha', 'shb', 'qnrf'])
parser.add_argument('--output_dir', type=str, default='saved_sha')
# training config
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--print_freq', type=int, default=200)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--weight_decay', type=float, default=5 * 1e-4)
parser.add_argument('--momentum', type=float, default=0.95)
parser.add_argument('--epochs', type=int, default=20000)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--best_pred', type=int, default=1e5)
parser.add_argument('--lr', type=float, default=1e-5)
# model config
parser.add_argument('--type_model', type=str, default='token', choices=['token', 'gap'])
parser.add_argument('--pre', type=str, default=None) # pretrained model dir
args = parser.parse_args()
return_args = parser.parse_args()