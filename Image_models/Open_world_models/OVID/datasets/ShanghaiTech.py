import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import os
import scipy as sp
import glob as gb

class ShanghaiTech(Dataset):
    def __init__(self, split: str, part: str, resize_val: bool=True):
        assert split in ['train', 'test']
        assert part in ['A', 'B']
        self.data_dir = "data/ShanghaiTech/part_{}/{}_data".format(part, split)
        self.resize_val = resize_val
        self.im_dir = os.path.join(self.data_dir,'images')
        self.anno_path = os.path.join(self.data_dir , "ground-truth")
        self.split = split
        self.img_paths = gb.glob(os.path.join(self.im_dir, "*.jpg"))
        self.img_names = [p.split("/")[-1].split(".")[0] for p in self.img_paths]
        self.gt_cnt = {}
        for im_name in self.img_names:
            assert os.path.exists(os.path.join(self.im_dir, f"{im_name}.jpg"))
            assert os.path.exists(os.path.join(self.anno_path, f"GT_{im_name}.mat"))
            with open(os.path.join(self.anno_path, f"GT_{im_name}.mat"), "rb") as f:
                mat = sp.io.loadmat(f)
                self.gt_cnt[im_name] = len(mat["image_info"][0][0][0][0][0])
        self.preprocess = transforms.Compose([transforms.Resize(384), transforms.ToTensor()])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        im_name = self.img_names[idx]
        im_path = os.path.join(self.im_dir, f"{im_name}.jpg")
        img = Image.open(im_path)
        if img.size[0] < img.size[1]:
            img = img.rotate(90, expand=True)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = self.preprocess(img) # [3, 384, 512]
        gt_cnt = self.gt_cnt[im_name] # 15
        return img, gt_cnt