import cv2
import numpy as np
from tqdm import tqdm
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter
from glob import glob
import os
import argparse
from multiprocessing import Pool

def gaussian_filter_density(img, points): # slow version
    img_shape = [img.shape[0],img.shape[1]]
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density
    leafsize = 2048
    tree = KDTree(points.copy(), leafsize=leafsize)
    distances, locations = tree.query(points, k=4)
    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1]) < img_shape[0] and int(pt[0]) < img_shape[1]:
            pt2d[int(pt[1]),int(pt[0])] = 1.
        else:
            continue
        if gt_count > 3:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = 15
        density += gaussian_filter(pt2d, sigma, mode='constant')
    return density

def gaussian_filter_density_fixed(img, points): # fast version
    img_shape = [img.shape[0], img.shape[1]]
    density = np.zeros(img_shape, dtype=np.float32) # [512, 766]
    gt_count = len(points) # 302
    if gt_count == 0:
        return density
    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1]) < img_shape[0] and int(pt[0]) < img_shape[1]:
            pt2d[int(pt[1]), int(pt[0])] = 1.
        else:
            continue
        sigma = 4
        density += gaussian_filter(pt2d, sigma, truncate=7 / sigma, mode='constant') # [512, 774]
    return density

def run(img_fn):
    img_ext = os.path.splitext(img_fn)[1]
    basename = os.path.basename(img_fn).replace(img_ext, '')
    gt_fn = img_fn.replace(img_ext, '.npy')
    dmap_fn = gt_fn.replace(basename, basename + '_dmap')
    if os.path.exists(dmap_fn):
        return
    img = cv2.imread(img_fn) # [512, 766, 3]
    gt = np.load(gt_fn) # [302, 2]
    dmap = gaussian_filter_density_fixed(img, gt) # [683, 512]
    np.save(dmap_fn, dmap)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/sta')
    parser.add_argument('--num_process', type=int, default=8)
    args = parser.parse_args()

    print('Generate density map for dataset:', args.input_dir.split('/')[-1])
    if not os.path.exists(args.input_dir):
        raise Exception("Path does not exist")
    img_fns = []
    for phase in ['train', 'val', 'test']:
        img_fns += glob(os.path.join(args.input_dir, phase, '*.jpg'))
    new_fns = []
    for fn in img_fns:
        if 'aug' in fn:
            continue
        new_fns.append(fn)
    img_fns = new_fns
    with Pool(args.num_process) as p:
        r = list(tqdm(p.imap(run, img_fns), total=len(img_fns)))