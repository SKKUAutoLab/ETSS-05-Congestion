import h5py
import scipy.io as io
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import argparse
import warnings
warnings.filterwarnings("ignore")

def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32) # [685, 1024]
    gt_count = np.count_nonzero(gt) # 321
    if gt_count == 0:
        return density
    y, x = np.nonzero(gt)
    pts = np.column_stack((x, y)) # [321, 2]
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=2048)
    distances, locations = tree.query(pts, k=4) # [321, 4], [321, 4]
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32) # [685, 1024]
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant') # [685, 1024]
    return density

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='datasets/ShanghaiTech')
    args = parser.parse_args()

    print('Process dataset:', args.input_dir.split('/')[-1])
    part_A_train = os.path.join(args.input_dir, 'part_A_final/train_data','images')
    part_A_test = os.path.join(args.input_dir, 'part_A_final/test_data','images')
    part_B_train = os.path.join(args.input_dir, 'part_B_final/train_data','images')
    part_B_test = os.path.join(args.input_dir, 'part_B_final/test_data','images')
    path_sets = [part_A_train,part_A_test]
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
    for img_path in img_paths:
        mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
        img = plt.imread(img_path)
        k = np.zeros((img.shape[0],img.shape[1])) # [685, 1024]
        gt = mat["image_info"][0, 0][0, 0][0]
        for i in range(0,len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                k[int(gt[i][1]), int(gt[i][0])] = 1
        k = gaussian_filter_density(k) # [685, 1024]
        with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'), 'w') as hf:
            hf['density'] = k
    path_sets = [part_B_train,part_B_test]
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
    for img_path in img_paths:
        mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
        img= plt.imread(img_path)
        k = np.zeros((img.shape[0],img.shape[1]))
        gt = mat["image_info"][0, 0][0, 0][0]
        for i in range(0,len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                k[int(gt[i][1]), int(gt[i][0])] = 1
        k = gaussian_filter(k, 15)
        with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'), 'w') as hf:
            hf['density'] = k