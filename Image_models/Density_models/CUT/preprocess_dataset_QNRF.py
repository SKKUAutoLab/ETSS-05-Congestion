from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
from glob import glob
import argparse
import scipy.spatial
import tqdm
import scipy.ndimage as ndimage
import scipy
import h5py

def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    distances, locations = tree.query(pts, k=4)
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
            if sigma > 15:
                sigma = 15
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.
        density += ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    return density

def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w * ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h * ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h * ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w * ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio

def generate_data(im_path, min_size, max_size):
    im = Image.open(im_path)
    im_w, im_h = im.size
    mat_path = im_path.replace('.jpg', '_ann.mat')
    points =  loadmat(mat_path)["annPoints"]
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    if rr != 1.0:
        im = im.resize((im_w, im_h), Image.LANCZOS)
        points = points * rr
    return im, points

def main(args):
    for phase in ['train', 'val']:
        sub_dir = os.path.join(args.output_dir, phase)
        sub_dir_img = os.path.join(sub_dir, 'images')
        sub_dir_pts = os.path.join(sub_dir, 'gt_points')
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
            os.makedirs(sub_dir_img)
            os.makedirs(sub_dir_pts)
        if phase == 'train':
            im_list = sorted(glob(os.path.join(args.input_dir, 'Train', "*.jpg")))
            sub_dir_den = os.path.join(sub_dir, 'gt_den')
            if not os.path.exists(sub_dir_den):
                os.makedirs(sub_dir_den)
        else:
            im_list = sorted(glob(os.path.join(args.input_dir, 'Test', "*.jpg")))
        for im_path in tqdm.tqdm(im_list):
            im, points = generate_data(im_path, 512, 1920)
            name = os.path.basename(im_path)
            if phase == 'train':
                w, h = im.size
                d = np.zeros((h, w))
                for j in range(len(points)):
                    point_x, point_y = points[j][0: 2].astype('int')
                    if point_y >= h or point_x >= w:
                        continue
                    d[point_y, point_x] = 1
                d = gaussian_filter_density(d)
                with h5py.File(os.path.join(sub_dir_den, '{}.h5'.format(name.replace('.jpg', ''))), 'w') as hf:
                    hf['density_map'] = d
                print(name, 'GT_num:', len(points), 'Density_sum: {:.2f}'.format(d.sum()))
            im_save_path = os.path.join(sub_dir_img, name)
            im.save(im_save_path, quality=100)
            gd_save_path = im_save_path.replace('jpg', 'npy').replace('images', 'gt_points')
            np.save(gd_save_path, points)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='qnrf')
    parser.add_argument('--input_dir', default='datasets/UCF-QNRF', type=str)
    parser.add_argument('--output_dir', default='processed_datasets/qnrf', type=str)
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    main(args)