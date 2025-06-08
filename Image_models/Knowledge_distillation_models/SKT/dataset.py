from torch.utils.data import Dataset
from image import load_ucf_ori_data, load_data
import random

class listDataset(Dataset):
    def __init__(self, root, transform=None, train=False, dataset='sha'):
        if train and (dataset == 'sha' or dataset == 'shb'):
            root = root * 4
        random.shuffle(root)
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.dataset = dataset

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.lines[index]
        if self.dataset == 'qnrf':
            img, target = load_ucf_ori_data(img_path)
        elif self.dataset == 'sha' or self.dataset == 'shb':
            img, target = load_data(img_path, self.train)
        else:
            print('This dataset does not exist')
            raise NotImplementedError
        if self.transform is not None:
            img = self.transform(img)
        return img, target