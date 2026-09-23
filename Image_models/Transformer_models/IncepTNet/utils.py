import logging
import os
import random
import shutil

import cv2
import h5py
import numpy as np
import torch
# from mmcv.utils import get_logger
from tqdm import tqdm

from image import load_data, load_data_test

# 新版本
from mmengine.logging import MMLogger

def get_root_logger(log_file=None, log_level='INFO'): 
    logger = MMLogger.get_instance(
        'IncepTNet',  
        logger_name='IncepTNet',
        log_level=log_level,
        log_file=log_file
    )
    
    import logging
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level)
    logger.setLevel(log_level)
    
    return logger

def save_results(input_img, gt_data, density_map, output_dir, fname='results.png'):
    density_map[density_map < 0] = 0

    gt_data = 255 * gt_data / np.max(gt_data)
    gt_data = gt_data[0][0]
    gt_data = gt_data.astype(np.uint8)
    gt_data = cv2.applyColorMap(gt_data, 2)

    density_map = 255 * density_map / np.max(density_map)
    density_map = density_map[0][0]
    density_map = density_map.astype(np.uint8)
    density_map = cv2.applyColorMap(density_map, 2)

    result_img = np.hstack((gt_data, density_map))

    cv2.imwrite(os.path.join('.', output_dir, fname).replace('.jpg', '.jpg'), result_img)


def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():
            param = torch.from_numpy(np.asarray(h5f[k]))
            v.copy_(param)


def save_checkpoint(state, visi, is_best, save_path, filename='checkpoint.pth'):
    torch.save(state, './' + str(save_path) + '/' + filename)
    if is_best:
        best_prec1 = str(state['best_prec1'])
        shutil.copyfile('./' + str(save_path) + '/' + filename, './' + str(save_path) + '/' + 'model_best_'+str(state['epoch'])+'_'+best_prec1[:5]+'.pth')

    for i in range(len(visi)):
        img = visi[i][0]
        output = visi[i][1]
        target = visi[i][2]
        fname = visi[i][3]
        save_results(img, target, output, str(save_path), fname[0])


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  

def pre_data_test(train_list, args, train):
    data_keys = {}
    count = 0
    for j in tqdm(range(len(train_list))):
        Img_path = train_list[j]
        fname = os.path.basename(Img_path)

        img = load_data_test(Img_path, args, train)

        blob = {}
        blob['img'] = img
        blob['fname'] = fname
        data_keys[count] = blob
        count += 1

    return data_keys


