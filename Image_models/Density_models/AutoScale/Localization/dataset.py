from torch.utils.data import Dataset
from image import load_data
import os

class listDataset(Dataset):
    def __init__(self, root, transform=None):
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.lines[index]
        fname = os.path.basename(img_path)
        img, target, kpoint, sigma_map= load_data(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return fname, img, target, kpoint, sigma_map