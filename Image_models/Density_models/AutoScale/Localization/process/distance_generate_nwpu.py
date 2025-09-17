import h5py
import numpy as np
import os
import glob
import scipy.spatial
import scipy
import math
import cv2
import argparse

def Distance_generate(im_data, k, lamda):
    size = im_data.shape
    new_im_data = cv2.resize(im_data, (lamda * size[1], lamda * size[0]), 0)
    distance = 1
    new_size = new_im_data.shape
    d_map = (np.zeros([new_size[0], new_size[1]]) + 255).astype(np.uint8)
    gt = np.nonzero(k)
    gt = lamda * gt
    if len(gt[0]) == 0:
        distance_map = np.zeros([int(new_size[0]), int(new_size[1])])
        distance_map[:, :] = 10
        return new_size, distance_map
    for o in range(0, len(gt[0])):
        x = np.max([1, math.floor(gt[0][o])])
        y = np.max([1, math.floor(gt[1][o])])
        if x >= new_size[0] or y >= new_size[1]:
            continue
        d_map[x][y] = d_map[x][y] - 255
    distance_map = cv2.distanceTransform(d_map, cv2.DIST_L2, 5)
    distance_map[(distance_map >= 0) & (distance_map < 1 * distance)] = 0
    distance_map[(distance_map >= 1 * distance) & (distance_map < 2 * distance)] = 1
    distance_map[(distance_map >= 2 * distance) & (distance_map < 3 * distance)] = 2
    distance_map[(distance_map >= 3 * distance) & (distance_map < 4 * distance)] = 3
    distance_map[(distance_map >= 4 * distance) & (distance_map < 5 * distance)] = 4
    distance_map[(distance_map >= 5 * distance) & (distance_map < 6 * distance)] = 5
    distance_map[(distance_map >= 6 * distance) & (distance_map < 8 * distance)] = 6
    distance_map[(distance_map >= 8 * distance) & (distance_map < 12 * distance)] = 7
    distance_map[(distance_map >= 12 * distance) & (distance_map < 18 * distance)] = 8
    distance_map[(distance_map >= 18 * distance) & (distance_map < 28 * distance)] = 9
    distance_map[(distance_map >= 28 * distance)] = 10
    return new_im_data, distance_map

def main(args):
    NWPU_Crowd_path = os.path.join(args.input_dir, 'images_2048/')
    path_sets = [NWPU_Crowd_path]
    if not os.path.exists(NWPU_Crowd_path.replace('images','gt_distance_map')):
        os.makedirs(NWPU_Crowd_path.replace('images','gt_distance_map'))
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
    img_paths.sort()
    for img_path in img_paths:
        img = cv2.imread(img_path)
        k = np.zeros((img.shape[0], img.shape[1]))
        mat_path = img_path.replace('images', 'gt_npydata').replace('jpg', 'npy')
        with open(mat_path, 'rb') as outfile:
            gt = np.load(outfile).tolist()
        for i in range(0, len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                k[int(gt[i][1]), int(gt[i][0])] = 1
        kpoint = k.copy()
        result = Distance_generate(img, k, 1)
        Distance_map = result[1]
        pts = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))
        leafsize = 2048
        if int(kpoint.sum()) > 1:
            tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
            distances, locations = tree.query(pts, k=2)
            sigma_map = np.zeros(kpoint.shape, dtype=np.float32)
            for i, pt in enumerate(pts):
                sigma = (distances[i][1]) / 2
                sigma_map[pt[1], pt[0]] = sigma
        elif int(kpoint.sum()) == 1:
            tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
            distances, locations = tree.query(pts, k=1)
            sigma_map = np.zeros(kpoint.shape, dtype=np.float32)
            for i, pt in enumerate(pts):
                sigma = (distances[i]) / 1
                sigma_map[pt[1], pt[0]] = sigma
        else:
            sigma_map = np.zeros(kpoint.shape, dtype=np.float32)
        with h5py.File(img_path.replace('images', 'gt_distance_map').replace('jpg', 'h5'), 'w') as hf:
            hf['distance_map'] = Distance_map
            hf['kpoint'] = kpoint
            hf['sigma_map'] = sigma_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='NWPU-Crowd')
    parser.add_argument('--input_dir', type=str, default='datasets/NWPU_localization')
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    main(args)