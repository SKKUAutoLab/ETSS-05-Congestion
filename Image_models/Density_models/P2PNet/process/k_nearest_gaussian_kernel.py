import warnings
warnings.filterwarnings("ignore")
import scipy.io as io
from scipy.ndimage.filters import gaussian_filter
import os
import glob
from matplotlib import pyplot as plt
import argparse
import numpy as np
import scipy

def save_gt_points_to_txt(gt_points, txt_path):
    with open(txt_path, 'w') as f:
        for point in gt_points:
            f.write(f"{point[0]} {point[1]}\n")

def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32) # [685, 1024]
    gt_count = np.count_nonzero(gt) # [321]
    if gt_count == 0:
        return density
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0]))) # [321, 2]
    leafsize = 2048
    tree = scipy.spatial.cKDTree(pts.copy(), leafsize=leafsize) # default: KDTree
    distances, locations = tree.query(pts, k=4)
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32) # [685, 1024]
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1 # [1]
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant') # [685, 1024]
    return density

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='sha', choices=['sha', 'shb'])
    parser.add_argument('--input_dir', type=str, default='datasets/ShanghaiTech/part_A')
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    part_A_train = os.path.join(args.input_dir, 'train_data', 'images')
    part_A_test = os.path.join(args.input_dir, 'test_data', 'images')
    path_sets = [part_A_train, part_A_test]
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
    for img_path in img_paths:
        mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG_', 'GT_IMG_'))
        img = plt.imread(img_path) # [685, 1024, 3]
        k = np.zeros((img.shape[0], img.shape[1])) # [685, 1024]
        gt = mat["image_info"][0, 0][0, 0][0] # [321, 2]
        save_gt_points_to_txt(gt, img_path.replace('.jpg', '.txt'))
        for i in range(0, len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                k[int(gt[i][1]), int(gt[i][0])] = 1
        k = gaussian_filter_density(k) # [685, 1024]
        np.save(img_path.replace('.jpg', '.npy').replace('images', 'ground-truth'), k)