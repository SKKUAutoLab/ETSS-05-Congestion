import warnings
warnings.filterwarnings("ignore")
import os
import cv2
import h5py
import numpy as np
import scipy.io
import scipy.spatial
from scipy.ndimage.filters import gaussian_filter
import argparse

def main(args):
    img_train_path = os.path.join(args.input_dir, 'Train/')
    gt_train_path = os.path.join(args.input_dir, 'Train/')
    img_test_path = os.path.join(args.input_dir, 'Test/')
    gt_test_path = os.path.join(args.input_dir, 'Test/')
    save_train_img_path = os.path.join(args.input_dir, 'train_data/images/')
    save_train_gt_path = os.path.join(args.input_dir, 'train_data/gt_distance_map/')
    save_test_img_path = os.path.join(args.input_dir, 'test_data/images/')
    save_test_gt_path = os.path.join(args.input_dir, 'test_data/gt_distance_map/')
    if not os.path.exists(save_train_img_path):
        os.makedirs(save_train_img_path)
    if not os.path.exists(save_train_gt_path):
        os.makedirs(save_train_gt_path)
    if not os.path.exists(save_train_img_path.replace('images', 'gt_show_density')):
        os.makedirs(save_train_img_path.replace('images', 'gt_show_density'))
    if not os.path.exists(save_test_img_path):
        os.makedirs(save_test_img_path)
    if not os.path.exists(save_test_gt_path):
        os.makedirs(save_test_gt_path)
    if not os.path.exists(save_test_img_path.replace('images', 'gt_show_density')):
        os.makedirs(save_test_img_path.replace('images', 'gt_show_density'))
    img_train = []
    gt_train = []
    img_test = []
    gt_test = []
    for file_name in os.listdir(img_train_path):
        if file_name.split('.')[1] == 'jpg':
            img_train.append(file_name)
    for file_name in os.listdir(gt_train_path):
        if file_name.split('.')[1] == 'mat':
            gt_train.append(file_name)
    for file_name in os.listdir(img_test_path):
        if file_name.split('.')[1] == 'jpg':
            img_test.append(file_name)
    for file_name in os.listdir(gt_test_path):
        if file_name.split('.')[1] == 'mat':
            gt_test.append(file_name)
    img_train.sort()
    gt_train.sort()
    img_test.sort()
    gt_test.sort()
    min_x = 640
    min_y = 480
    x = []
    y = []
    for k in range(len(img_train)):
        Img_data = cv2.imread(img_train_path + img_train[k])
        Gt_data = scipy.io.loadmat(gt_train_path + gt_train[k])
        rate = 1
        if Img_data.shape[1]>=Img_data.shape[0] and Img_data.shape[1] >= 1024:
            rate = 1024.0 / Img_data.shape[1]
        if Img_data.shape[0]>=Img_data.shape[1] and Img_data.shape[0] >= 1024:
            rate = 1024.0 / Img_data.shape[0]
        Img_data = cv2.resize(Img_data,(0,0),fx=rate,fy=rate)
        x.append(Img_data.shape[1])
        y.append(Img_data.shape[0])
        Gt_data = Gt_data['annPoints']
        Gt_data = Gt_data * rate
        density_map = np.zeros((Img_data.shape[0], Img_data.shape[1]))
        for count in range(0, len(Gt_data)):
            if int(Gt_data[count][1]) < Img_data.shape[0] and int(Gt_data[count][0]) < Img_data.shape[1]:
                density_map[int(Gt_data[count][1]), int(Gt_data[count][0])] = 1
        kpoint = density_map.copy()
        density_map = gaussian_filter(density_map, 6)
        new_img_path = (save_train_img_path + img_train[k])
        gt_show_path = new_img_path.replace('images', 'gt_show_density')
        h5_path = save_train_gt_path + img_train[k].replace('.jpg','.h5')
        pts = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))
        leafsize = 2048
        tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
        distances, locations = tree.query(pts, k=2)
        sigma_map = np.zeros(kpoint.shape, dtype=np.float32)
        for i, pt in enumerate(pts):
            sigma = (distances[i][1]) / 2
            sigma_map[pt[1], pt[0]] = sigma
        with h5py.File(h5_path, 'w') as hf:
            hf['density_map'] = density_map
            hf['kpoint'] = kpoint
            hf['sigma_map'] = sigma_map
        cv2.imwrite(new_img_path, Img_data)
        density_map = density_map / np.max(density_map) * 255
        density_map = density_map.astype(np.uint8)
        density_map = cv2.applyColorMap(density_map,2)
        cv2.imwrite(gt_show_path, density_map)
    for k in range(len(img_test)):
        Img_data = cv2.imread(img_test_path + img_test[k])
        Gt_data = scipy.io.loadmat(gt_test_path + gt_test[k])
        rate = 1
        if Img_data.shape[1] > Img_data.shape[0] and Img_data.shape[1] >=1024:
            rate = 1024.0 / Img_data.shape[1]
        if Img_data.shape[0] > Img_data.shape[1] and Img_data.shape[0] >=1024:
            rate = 1024.0 / Img_data.shape[0]
        Img_data = cv2.resize(Img_data, (0, 0), fx=rate, fy=rate)
        if Img_data.shape[0]<min_y:
            min_y = Img_data.shape[0]
        if Img_data.shape[1]<min_x:
            min_x = Img_data.shape[1]
        Gt_data = Gt_data['annPoints']
        Gt_data = Gt_data * rate
        density_map = np.zeros((Img_data.shape[0], Img_data.shape[1]))
        for count in range(0, len(Gt_data)):
            if int(Gt_data[count][1]) < Img_data.shape[0] and int(Gt_data[count][0]) < Img_data.shape[1]:
                density_map[int(Gt_data[count][1]), int(Gt_data[count][0])] = 1
        kpoint = density_map.copy()
        density_map = gaussian_filter(density_map, 6)
        new_img_path = (save_test_img_path + img_test[k])
        gt_show_path = new_img_path.replace('images','gt_show_density')
        h5_path = save_test_gt_path + img_test[k].replace('.jpg','.h5')
        pts = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))
        leafsize = 2048
        tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
        distances, locations = tree.query(pts, k=2)
        sigma_map = np.zeros(kpoint.shape, dtype=np.float32)
        for i, pt in enumerate(pts):
            sigma = (distances[i][1]) / 2
            sigma_map[pt[1], pt[0]] = sigma
        with h5py.File(h5_path, 'w') as hf:
            hf['density_map'] = density_map
            hf['kpoint'] = kpoint
            hf['sigma_map'] = sigma_map
        cv2.imwrite(new_img_path, Img_data)
        density_map = density_map / np.max(density_map) * 255
        density_map = density_map.astype(np.uint8)
        density_map = cv2.applyColorMap(density_map,2)
        cv2.imwrite(gt_show_path, density_map)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='UCF-QNRF')
    parser.add_argument('--input_dir', type=str, default='datasets/UCF-QNRF')
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    main(args)