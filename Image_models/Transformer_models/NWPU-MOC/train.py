import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import torch
from importlib import import_module
from trainer import Trainer
import argparse
from config import cfg

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

def main(args):
    setup_seed(args.seed)
    if len(cfg.GPU_ID) > 1:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.GPU_ID).strip("[").strip("]")
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(cfg.GPU_ID[0])
    datasetting = import_module(f'datasets.setting.{cfg.DATASET}')
    cfg_data = datasetting.cfg_data
    pwd = os.path.split(os.path.realpath(__file__))[0]
    cc_trainer = Trainer(cfg_data, pwd)
    cc_trainer.forward()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='NWPU-MOC')
    parser.add_argument('--seed', type=int, default=42) # 3035
    args = parser.parse_args()

    print('Training dataset:', args. type_dataset)
    main(args)