import os
import json
import h5py
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2

def find_dis(points):
    if len(points) < 2:
        return np.array([4] * len(points))
    square = np.sum(points * points, axis=1)
    dis = np.sqrt(np.maximum(square[:, None] - 2 * np.matmul(points, points.T) + square[None, :], 0.0))
    dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
    return dis.flatten()

def generate_density_map(img_shape, points):
    h, w = img_shape
    density = np.zeros((h, w), dtype=np.float32)
    if len(points) == 0:
        return np.stack([density, density], axis=0)
    sigmas = find_dis(points)
    for (x, y), sigma in zip(points, sigmas):
        if 0 <= int(y) < h and 0 <= int(x) < w:
            tmp = np.zeros((h, w), dtype=np.float32)
            tmp[int(y), int(x)] = 1
            tmp = gaussian_filter(tmp, sigma=sigma, mode='constant')
            density += tmp
    density = np.stack([density, density], axis=0)
    return density

def resize_all(img, den, mask, size=(256, 256)):
    h, w = size
    img = cv2.resize(np.array(img), (w, h), interpolation=cv2.INTER_CUBIC)
    den = np.array([cv2.resize(den_ch, (w, h), interpolation=cv2.INTER_CUBIC) for den_ch in den])
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return img, den, mask

class CrowdDataset(Dataset):
    def __init__(self, root_dir, split="train", resize=(256, 256)):
        self.img_dir = os.path.join(root_dir, "images")
        self.ann_dir = os.path.join(root_dir, "annotations")
        self.resize = resize
        split_file = os.path.join(self.ann_dir, f"{split}.json")
        with open(split_file, "r") as f:
            self.img_list = json.load(f)
        anno_file = os.path.join(self.ann_dir, "annotations.json")
        with open(anno_file, "r") as f:
            self.annotations = json.load(f)
        mask_file = os.path.join(self.ann_dir, "mask.h5")
        with h5py.File(mask_file, "r") as f:
            self.mask = np.array(f["mask"], dtype=np.uint8)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        points_list = self.annotations.get(img_name, [])
        coords = []
        for group in points_list:
            xs = group["x"]
            ys = group["y"]
            coords.extend(list(zip(xs, ys)))
        coords = np.array(coords, dtype=np.float32)
        den = generate_density_map(img.size[::-1], coords)
        img, den, mask = resize_all(img, den, self.mask, size=self.resize)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        den = torch.from_numpy(den).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return {"img": img, "den": den, "mask": mask}

class DataLoader:
    def __init__(self, args):
        self.dataset_root = args.input_dir
        self.batch_size = args.batch_size
        self.num_workers = getattr(args, "nThreads", 1)

    def get_train_loader(self):
        train_set = CrowdDataset(self.dataset_root, split="train", resize=(256, 256))
        return TorchDataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def get_val_loader(self):
        val_set = CrowdDataset(self.dataset_root, split="val", resize=(256, 256))
        return TorchDataLoader(val_set, batch_size=1, shuffle=False, num_workers=self.num_workers)

    def get_test_loader(self):
        test_set = CrowdDataset(self.dataset_root, split="test", resize=(256, 256))
        return TorchDataLoader(test_set, batch_size=1, shuffle=False, num_workers=self.num_workers)