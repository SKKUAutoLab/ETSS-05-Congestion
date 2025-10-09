import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2

class SHHA(Dataset):
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False, type_dataset='sha'):
        self.root_path = data_root
        if train:
            if type_dataset == 'sha':
                self.img_list_file = os.path.join(data_root, "train_sha.txt")
            elif type_dataset == 'shb':
                self.img_list_file = os.path.join(data_root, "train_shb.txt")
            else:
                print('This dataset does not exist')
                raise NotImplementedError
        else:
            if type_dataset == 'sha':
                self.img_list_file = os.path.join(data_root, "test_sha.txt")
            elif type_dataset == 'shb':
                self.img_list_file = os.path.join(data_root, "test_shb.txt")
            else:
                print('This dataset does not exist')
                raise NotImplementedError
        self.img_map = {}
        self.img_list = []
        with open(self.img_list_file) as fin:
            for line in fin:
                if len(line) < 2:
                    continue
                line = line.strip().split()
                self.img_map[os.path.join(self.root_path, line[0].strip())] = os.path.join(self.root_path, line[1].strip())
        self.img_list = sorted(list(self.img_map.keys()))
        self.nSamples = len(self.img_list)
        self.transform = transform
        self.train = train
        self.patch = patch
        self.flip = flip

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.img_list[index]
        gt_path = self.img_map[img_path]
        img, point = load_data(img_path, gt_path) # [460, 370, 3], [74, 2]
        if self.transform is not None:
            img = self.transform(img) # [3, 464, 370]
        if self.train:
            scale_range = [0.7, 1.3]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)
            if scale * min_size > 128:
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0) # [3, 440, 358]
                point *= scale
        if self.train and self.patch:
            img, point = random_crop(img, point) # [4, 3, 128, 128], [2, 2]
            for i, _ in enumerate(point):
                point[i] = torch.Tensor(point[i])
        if random.random() > 0.5 and self.train and self.flip:
            img = torch.Tensor(img[:, :, :, ::-1].copy())
            for i, _ in enumerate(point):
                point[i][:, 0] = 128 - point[i][:, 0]
        if not self.train:
            point = [point]
        img = torch.Tensor(img)
        target = [{} for i in range(len(point))]
        for i, _ in enumerate(point):
            target[i]['point'] = torch.Tensor(point[i])
            image_id = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])
            image_id = torch.Tensor([image_id]).long()
            target[i]['image_id'] = image_id
            target[i]['labels'] = torch.ones([point[i].shape[0]]).long()
        return img, target

def load_data(img_path, gt_path):
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    points = []
    with open(gt_path) as f_label:
        for line in f_label:
            x = float(line.strip().split(' ')[0])
            y = float(line.strip().split(' ')[1])
            points.append([x, y])
    return img, np.array(points)

def random_crop(img, den, num_patch=4): # [3, 449, 358], [74, 2]
    half_h = 128
    half_w = 128
    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
    result_den = []
    for i in range(num_patch):
        start_h = random.randint(0, img.size(1) - half_h)
        start_w = random.randint(0, img.size(2) - half_w)
        end_h = start_h + half_h
        end_w = start_w + half_w
        result_img[i] = img[:, start_h:end_h, start_w:end_w]
        idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
        record_den = den[idx]
        record_den[:, 0] -= start_w
        record_den[:, 1] -= start_h
        result_den.append(record_den)
    return result_img, result_den