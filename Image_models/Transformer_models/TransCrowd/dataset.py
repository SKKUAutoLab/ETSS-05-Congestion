import torch
from torch.utils.data import Dataset
from PIL import Image
import random

class listDataset(Dataset):
    def __init__(self, root, shape=None, transform=None, train=False):
        if train:
            random.shuffle(root)
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        fname = self.lines[index]['fname']
        img = self.lines[index]['img']
        gt_count = self.lines[index]['gt_count']
        if self.train == True:
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        gt_count = gt_count.copy()
        img = img.copy()
        if self.train == True:
            if self.transform is not None:
                img = self.transform(img) # [3, 384, 384]
            return fname, img, gt_count
        else:
            if self.transform is not None:
                img = self.transform(img) # [3, 768, 1152]
            width, height = img.shape[2], img.shape[1]
            m = int(width / 384)
            n = int(height / 384)
            for i in range(0, m):
                for j in range(0, n):
                    if i == 0 and j == 0:
                        img_return = img[:, j * 384: 384 * (j + 1), i * 384:(i + 1) * 384].cuda().unsqueeze(0) # [1, 3, 384, 384]
                    else:
                        crop_img = img[:, j * 384: 384 * (j + 1), i * 384:(i + 1) * 384].cuda().unsqueeze(0) # [1, 3, 384, 384]
                        img_return = torch.cat([img_return, crop_img], 0).cuda() # [1, 3, 384, 384]
            return fname, img_return, gt_count