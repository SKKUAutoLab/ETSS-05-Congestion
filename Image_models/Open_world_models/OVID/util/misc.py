from typing import List
import argparse
import torch
from torch import Tensor
import numpy as np

def _max_by_axis_pad(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    block = 128
    for i in range(2):
        maxes[i+1] = ((maxes[i+1] - 1) // block + 1) * block
    return maxes

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis_pad([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        for img, pad_img in zip(tensor_list, tensor):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    else:
        raise ValueError('not supported')
    return tensor

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    for i in range(1,4):
        if i==2:
            continue
        batch[i] = torch.stack(batch[i], dim=0)
    return tuple(batch)

def sliding_window(image, window_size = (384, 384), stride = 128):
    if isinstance(image, torch.Tensor):
        if image.shape[0] == 1:
            image = image.squeeze(0)
        image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()
    image = np.pad(image, ((0, 0), (0, stride - image.shape[1] % stride), (0, 0)), 'constant')
    h, w, _ = image.shape
    assert h == 384, "FSC-147 assume image height is 384."
    patches = []
    intervals = []
    for i in range(0, w - window_size[1] + 1, stride):
        patch = image[:, i:i + window_size[1], :]
        patches.append(patch)
        intervals.append([i, i + window_size[1]])
    return np.array(patches).transpose(0, 3, 1, 2), np.array(intervals)