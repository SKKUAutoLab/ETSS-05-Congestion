import argparse

parser = argparse.ArgumentParser()
# general config
parser.add_argument('--type_dataset', type=str, default='sha')
parser.add_argument('--print_freq', type=int, default=30)
parser.add_argument('--pre', type=str, default='ckpts/ShanghaiA_localization/model_best.pth')
parser.add_argument('--seed', type=int, default=1)
# testing config
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--area_threshold', type=float, default=0.02)
args = parser.parse_args()