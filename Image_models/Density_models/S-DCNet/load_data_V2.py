import os
from skimage import io
from torch.utils.data import Dataset
import glob
import scipy.io as sio
import torch
import torch.nn.functional as F

class myDataset(Dataset):
    def __init__(self, img_dir,tar_dir, rgb_dir,transform=None,if_test = False, IF_loadmem=False):
        self.IF_loadmem = IF_loadmem
        self.IF_loadFinished = False
        self.image_mem = []
        self.target_mem = []
        self.img_dir = img_dir
        self.tar_dir = tar_dir
        self.transform = transform
        mat = sio.loadmat(rgb_dir)
        self.rgb = mat['rgbMean'].reshape(1, 1, 3)
        img_name = os.path.join(self.img_dir,'*.jpg')
        self.filelist =  glob.glob(img_name)
        self.dataset_len = len(self.filelist)
        self.if_test = if_test

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if (not self.IF_loadmem) or (not self.IF_loadFinished): 
            img_name =self.filelist[idx]
            image = io.imread(img_name)
            image = image / 255. - self.rgb
            filepath, tempfilename = os.path.split(img_name)
            name, extension = os.path.splitext(tempfilename)
            mat_dir = os.path.join( self.tar_dir, '%s.mat' % name)
            mat = sio.loadmat(mat_dir)
            if self.IF_loadmem:
                self.image_mem.append(image)
                self.target_mem.append(mat)
                if len(self.image_mem) == self.dataset_len:
                    self.IF_loadFinished = True
        else:
            image = self.image_mem[idx]
            mat = self.target_mem[idx]
        if not self.if_test:
            target = mat['crop_gtdens']
            sample = {'image': image, 'target': target}
            if self.transform:
                sample = self.transform(sample)
            sample['image'], sample['target'] = get_pad(sample['image'],DIV=64), get_pad(sample['target'], DIV=64)
        else:
            target = mat['all_num']
            sample = {'image': image, 'target': target}
            if self.transform:
                sample = self.transform(sample)
            sample['density_map'] = torch.from_numpy(mat['density_map'])
            sample['image'], sample['density_map'] = get_pad(sample['image'], DIV=64),get_pad(sample['density_map'], DIV=64)
        return sample

class ToTensor(object):
    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'target': torch.from_numpy(target)}

def get_pad(inputs,DIV=64):
    h, w = inputs.size()[-2:]
    ph, pw = (DIV - h % DIV), (DIV - w % DIV)
    if (ph != DIV) or (pw != DIV):
        tmp_pad = [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2]
        inputs = F.pad(inputs, tmp_pad)
    return inputs