import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
import os
import sys
sys.path.append('..')
from datasets.den_dataset import DensityMapDataset
from utils.misc import random_crop, get_padding

class DenClsDataset(DensityMapDataset):
    def collate(batch):
        transposed_batch = list(zip(*batch))
        images1 = torch.stack(transposed_batch[0], 0) # [16, 3, 320, 320]
        images2 = torch.stack(transposed_batch[1], 0) # [16, 3, 320, 320]
        points = transposed_batch[2] # [87, 2] * len(16)
        dmaps = torch.stack(transposed_batch[3], 0) # [16, 1, 320, 320]
        bmaps = torch.stack(transposed_batch[4], 0) # [16, 1, 20, 20]
        return images1, images2, (points, dmaps, bmaps)

    def __init__(self, root, crop_size, downsample, method, is_grey, unit_size, pre_resize=1, roi_map_path=None, gt_dir=None, gen_root=None):
        super().__init__(root, crop_size, downsample, method, is_grey, unit_size, pre_resize, roi_map_path, gt_dir, gen_root)
        self.more_transform = T.Compose([T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.1)], p=0.8),
                                         T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=1)], p=0.5), T.RandomAdjustSharpness(sharpness_factor=5, p=0.5),
                                         T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def __getitem__(self, index):
        img_fn = self.img_fns[index]
        img, img_ext = self._load_img(img_fn)
        basename = img_fn.split('/')[-1].split('.')[0]
        if img_fn.startswith(self.root):
            gt_fn = img_fn.replace(img_ext, '.npy')
            if basename.endswith('_aug'):
                gt_fn = gt_fn.replace('_aug', '')
            elif basename.endswith('_aug2'):
                gt_fn = gt_fn.replace('_aug2', '')
        else:
            basename = basename[:-2]
            gt_fn = os.path.join(self.root, 'train', basename + '.npy')
        gt = self._load_gt(gt_fn)
        if self.method == 'train':
            if self.gt_dir is None:
                dmap_fn = gt_fn.replace(basename, basename + '_dmap')
            else:
                dmap_fn = os.path.join(self.gt_dir, basename + '.npy')
            dmap = self._load_dmap(dmap_fn)
            img1, img2, gt, dmap = self._train_transform(img, gt, dmap) # [3, 320, 320], [3, 320, 320], [56, 2], [1, 320, 320]
            bmap_orig = dmap.clone().reshape(1, dmap.shape[1]//16, 16, dmap.shape[2]//16, 16).sum(dim=(2, 4)) # [1, 20, 20]
            bmap = (bmap_orig > 0).float() # [1, 20, 20]
            return img1, img2, gt, dmap, bmap
        elif self.method in ['val', 'test']:
            return self._val_transform(img, gt, basename)

    def _train_transform(self, img, gt, dmap):
        w, h = img.size
        assert len(gt) >= 0
        dmap = torch.from_numpy(dmap).unsqueeze(0) # [1, 620, 620]
        if random.random() > 0.88:
            img = img.convert('L').convert('RGB')
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
        img1 = self.transform(img) # [3, 320, 320]
        img2 = self.more_transform(img) # [3, 320, 320]
        gt = torch.from_numpy(gt.copy()).float() # [56, 2]
        dmap = dmap.float() # [1, 320, 320]
        return img1, img2, gt, dmap

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
        img1 = self.transform(img) # [3, 768, 1024]
        img2 = self.more_transform(img) # [3, 768, 1024]
        gt = torch.from_numpy(gt.copy()).float() # [298, 2]
        return img1, img2, gt, name, padding