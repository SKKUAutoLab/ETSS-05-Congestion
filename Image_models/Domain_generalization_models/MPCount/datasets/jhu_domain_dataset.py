import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image
import random
import os
import sys
sys.path.append('..')
from utils.misc import random_crop, get_padding

class JHUDomainDataset(Dataset):
    def collate(batch):
        transposed_batch = list(zip(*batch))
        images = torch.stack(transposed_batch[0], 0)
        points = transposed_batch[1]
        dmaps = torch.stack(transposed_batch[2], 0)
        return images, (points, dmaps)

    def __init__(self, root, domain_label, crop_size, domain_type, domain, downsample, method, is_grey=False, unit_size=0, pre_resize=1):
        self.root = root
        self.domain_label = domain_label
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size
        self.downsample = downsample
        self.method = method
        self.is_grey = is_grey
        self.unit_size = unit_size
        self.pre_resize = pre_resize
        phase_dict = {'train': 'train', 'val': 'val', 'test': 'val'}
        img_txt = os.path.join(self.root, 'domains', f'{domain_label}_{phase_dict[method]}.txt')
        with open(img_txt, 'r') as f:
            self.img_fns = f.readlines()
        self.img_fns = [img_fn.strip() for img_fn in self.img_fns]
        if self.method == 'train':
            self.transform = T.Compose([T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        else:
            self.transform = T.Compose([T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        
    def __len__(self):
        return len(self.img_fns)

    def _load_img(self, img_fn):
        img = Image.open(img_fn).convert('RGB')
        img_ext = os.path.splitext(img_fn)[1]
        return img, img_ext

    def _load_gt(self, gt_fn):
        gt = np.load(gt_fn)
        return gt
    
    def _load_dmap(self, dmap_fn):
        dmap = np.load(dmap_fn)
        return dmap

    def __getitem__(self, index):
        img_fn = self.img_fns[index]
        img, img_ext = self._load_img(img_fn)
        basename = img_fn.split('/')[-1].split('.')[0]
        if img_fn.startswith(self.root):
            gt_fn = img_fn.replace(img_ext, '.npy')
        else:
            basename = basename[:-2]
            gt_fn = os.path.join(self.root, 'train', basename + '.npy')
        gt = self._load_gt(gt_fn)
        if self.method == 'train':
            dmap_fn = gt_fn.replace(basename, basename + '_dmap')
            dmap = self._load_dmap(dmap_fn)
            return tuple(self._train_transform(img, gt, dmap))
        elif self.method in ['val', 'test']:
            return tuple(self._val_transform(img, gt, basename))

    def _train_transform(self, img, gt, dmap):
        w, h = img.size
        assert len(gt) >= 0
        dmap = torch.from_numpy(dmap).unsqueeze(0)
        if random.random() > 0.88:
            img = img.convert('L').convert('RGB')
        factor = self.pre_resize * (random.random() * 0.5 + 0.75)
        if factor != 1.0:
            new_w = (int)(w * factor)
            new_h = (int)(h * factor)
            w = new_w
            h = new_h
            img = img.resize((w, h))
            dmap_sum = dmap.sum()
            dmap = F.resize(dmap, (h, w))
            new_dmap_sum = dmap.sum()
            dmap = dmap * dmap_sum / new_dmap_sum
            gt = gt * factor
        st_size = 1.0 * min(w, h)
        if st_size < min(self.crop_size[0], self.crop_size[1]):
            padding, h, w = get_padding(h, w, self.crop_size[0], self.crop_size[1])
            left, top, _, _ = padding
            img = F.pad(img, padding)
            dmap = F.pad(dmap, padding)
            if len(gt) > 0:
                gt = gt + [left, top]
        i, j = random_crop(h, w, self.crop_size[0], self.crop_size[1])
        h, w = self.crop_size[0], self.crop_size[1]
        img = F.crop(img, i, j, h, w)
        h, w = self.crop_size[0], self.crop_size[1]
        dmap = F.crop(dmap, i, j, h, w)
        h, w = self.crop_size[0], self.crop_size[1]
        if len(gt) > 0:
            gt = gt - [j, i]
            idx_mask = (gt[:, 0] >= 0) * (gt[:, 0] <= w) * (gt[:, 1] >= 0) * (gt[:, 1] <= h)
            gt = gt[idx_mask]
        else:
            gt = np.empty([0, 2])
        down_w = w // self.downsample
        down_h = h // self.downsample
        dmap = dmap.reshape([1, down_h, self.downsample, down_w, self.downsample]).sum(dim=(2, 4))
        if len(gt) > 0:
            gt = gt / self.downsample
        if random.random() > 0.5:
            img = F.hflip(img)
            dmap = F.hflip(dmap)
            if len(gt) > 0:
                gt[:, 0] = w - gt[:, 0]
        img = self.transform(img)
        gt = torch.from_numpy(gt.copy()).float()
        dmap = dmap.float()
        return img, gt, dmap
    
    def _val_transform(self, img, gt, name):
        if self.pre_resize != 1:
            img = img.resize((int(img.size[0] * self.pre_resize), int(img.size[1] * self.pre_resize)))
        if self.unit_size is not None and self.unit_size > 0:
            w, h = img.size
            new_w = (w // self.unit_size + 1) * self.unit_size if w % self.unit_size != 0 else w
            new_h = (h // self.unit_size + 1) * self.unit_size if h % self.unit_size != 0 else h
            padding, h, w = get_padding(h, w, new_h, new_w)
            left, top, _, _ = padding
            img = F.pad(img, padding)
            if len(gt) > 0:
                gt = gt + [left, top]
        else:
            padding = (0, 0, 0, 0)
        gt = gt / self.downsample
        img = self.transform(img)
        gt = torch.from_numpy(gt.copy()).float()
        return img, gt, name, padding