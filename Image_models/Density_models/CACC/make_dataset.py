import scipy.io as io
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import argparse
import os
import numpy as np
import h5py

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='datasets/ShanghaiTech/part_B')
    args = parser.parse_args()

    print('Process dataset:', args.input_dir.split('/')[-1])
    part_B_train = os.path.join(args.input_dir, 'train_data/images')
    part_B_test = os.path.join(args.input_dir, 'test_data/images')
    path_sets = [part_B_train, part_B_test]
    img_paths  = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
    for  img_path  in img_paths:
        mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG_', 'GT_IMG_'))
        img = plt.imread(img_path)
        k = np.zeros((img.shape[0], img.shape[1]))
        gt = mat["image_info"][0, 0][0, 0][0]
        for i in range(0, len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                k[int(gt[i][1]), int(gt[i][0])] = 1
        k = gaussian_filter(k, 15)
        with h5py.File(img_path.replace('.jpg','.h5').replace('images', 'ground-truth'), 'w') as hf:
            hf['density'] = k