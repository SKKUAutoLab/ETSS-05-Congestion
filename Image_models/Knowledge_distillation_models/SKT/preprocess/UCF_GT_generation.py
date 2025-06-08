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

def gaussian_filter_density(gt, threshold=30):
    density = np.zeros(gt.shape, dtype=np.float32) # [2832, 4256]
    gt_count = np.count_nonzero(gt) # 433
    if gt_count == 0:
        return density
    y, x = np.nonzero(gt)
    pts = np.column_stack((x, y)) # [433, 2]
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=2048)
    distances, locations = tree.query(pts, k=2) # [433, 2], [433, 2]
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32) # [2832, 4256]
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = distances[i][1]
            sigma = min(sigma, threshold)
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant') # [2832, 4256]
    return density

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='datasets/UCF-QNRF')
    args = parser.parse_args()

    print('Process dataset:', args.input_dir.split('/')[-1])
    train_path = os.path.join(args.input_dir, 'Train')
    test_path = os.path.join(args.input_dir, 'Test')
    for path in [train_path, test_path]:
        paths = glob.glob(os.path.join(path, '*.jpg'))
        paths.sort()
        for img_path in paths:
            img = plt.imread(img_path) # [2832, 4256, 3]
            (name, _) = os.path.splitext(img_path)
            mat = io.loadmat(name+'_ann.mat')
            gt = mat['annPoints'] # [433, 2]
            k = np.zeros((img.shape[0], img.shape[1])) # [2832, 4256]
            for i in range(0, len(gt)):
                if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                    k[int(gt[i][1]), int(gt[i][0])] = 1
            k = gaussian_filter_density(k)
            with h5py.File(img_path.replace('.jpg', '.h5'), 'w') as hf:
                hf['density'] = k