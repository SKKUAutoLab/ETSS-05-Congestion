import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from importlib import import_module
from trainer import Trainer
import argparse

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='SHHA', choices=['SHHA', 'SHHB', 'QNRF', 'NWPU', 'FDST', 'JHU'])
    parser.add_argument('--output_dir', type=str, default='saved_sha')
    parser.add_argument('--vis_dir', type=str, default='vis_sha')
    parser.add_argument('--seed', type=int, default=3035)
    args = parser.parse_args()

    print('Training dataset:', args.type_dataset)
    setup_seed(args.seed)
    datasetting = import_module(f'datasets.setting.{args.type_dataset}')
    cc_trainer = Trainer(datasetting.cfg_data, args)
    cc_trainer.forward()