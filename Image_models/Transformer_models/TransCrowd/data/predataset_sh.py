import glob
import os
import cv2
import h5py
import numpy as np
import scipy.io as io
import random
import argparse
import warnings
warnings.filterwarnings("ignore")

def main(args):
    part_A_train = os.path.join(args.input_dir, 'part_A_final/train_data', 'images')
    part_A_test = os.path.join(args.input_dir, 'part_A_final/test_data', 'images')
    part_B_train = os.path.join(args.input_dir, 'part_B_final/train_data', 'images')
    part_B_test = os.path.join(args.input_dir, 'part_B_final/test_data', 'images')
    path_sets = [part_A_train, part_A_test, part_B_train, part_B_test]
    # process partA
    if not os.path.exists(part_A_train.replace('images', 'gt_density_map_crop')):
        os.makedirs(part_A_train.replace('images', 'gt_density_map_crop'))
    if not os.path.exists(part_A_test.replace('images', 'gt_density_map_crop')):
        os.makedirs(part_A_test.replace('images', 'gt_density_map_crop'))
    if not os.path.exists(part_A_train.replace('images', 'images_crop')):
        os.makedirs(part_A_train.replace('images', 'images_crop'))
    if not os.path.exists(part_A_test.replace('images', 'images_crop')):
        os.makedirs(part_A_test.replace('images', 'images_crop'))
    # process partB
    if not os.path.exists(part_B_train.replace('images', 'gt_density_map_crop')):
        os.makedirs(part_B_train.replace('images', 'gt_density_map_crop'))
    if not os.path.exists(part_B_test.replace('images', 'gt_density_map_crop')):
        os.makedirs(part_B_test.replace('images', 'gt_density_map_crop'))
    if not os.path.exists(part_B_train.replace('images', 'images_crop')):
        os.makedirs(part_B_train.replace('images', 'images_crop'))
    if not os.path.exists(part_B_test.replace('images', 'images_crop')):
        os.makedirs(part_B_test.replace('images', 'images_crop'))
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
    img_paths.sort()
    np.random.seed(0)
    random.seed(0)
    for img_path in img_paths:
        Img_data = cv2.imread(img_path) # [704, 1024, 3]
        mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
        Gt_data = mat["image_info"][0][0][0][0][0] # [172, 2]
        rate_1 = 1
        rate_2 = 1
        if Img_data.shape[1] >= Img_data.shape[0]:
            rate_1 = 1152.0 / Img_data.shape[1]
            rate_2 = 768 / Img_data.shape[0]
            Img_data = cv2.resize(Img_data, (0, 0), fx=rate_1, fy=rate_2) # [768, 1152, 3]
            Gt_data[:, 0] = Gt_data[:, 0] * rate_1
            Gt_data[:, 1] = Gt_data[:, 1] * rate_2
        elif Img_data.shape[0] > Img_data.shape[1]:  # 前面的大
            rate_1 = 1152.0 / Img_data.shape[0]
            rate_2 = 768.0 / Img_data.shape[1]
            Img_data = cv2.resize(Img_data, (0, 0), fx=rate_2, fy=rate_1) # [1152, 768, 3]
            Gt_data[:, 0] = Gt_data[:, 0] * rate_2
            Gt_data[:, 1] = Gt_data[:, 1] * rate_1
        kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1])) # [768, 1152]
        for i in range(0, len(Gt_data)):
            if int(Gt_data[i][1]) < Img_data.shape[0] and int(Gt_data[i][0]) < Img_data.shape[1]:
                kpoint[int(Gt_data[i][1]), int(Gt_data[i][0])] = 1
        height, width = Img_data.shape[0], Img_data.shape[1]
        m = int(width / 384)
        n = int(height / 384)
        fname = img_path.split('/')[-1]
        root_path = img_path.split('IMG_')[0].replace('images', 'images_crop')
        kpoint = kpoint.copy()
        if root_path.split('/')[-3] == 'train_data':
            for i in range(0, m):
                for j in range(0, n):
                    crop_img = Img_data[j * 384: 384 * (j + 1), i * 384:(i + 1) * 384] # [384, 384, 3]
                    crop_kpoint = kpoint[j * 384: 384 * (j + 1), i * 384:(i + 1) * 384] # [384, 384]
                    gt_count = np.sum(crop_kpoint)
                    save_fname = str(i) + str(j) + str('_') + fname
                    save_path = root_path + save_fname
                    h5_path = save_path.replace('.jpg', '.h5').replace('images', 'gt_density_map')
                    with h5py.File(h5_path, 'w') as hf:
                        hf['gt_count'] = gt_count
                    cv2.imwrite(save_path, crop_img)
        else:
            img_path = img_path.replace('images', 'images_crop')
            cv2.imwrite(img_path, Img_data)
            gt_count = np.sum(kpoint)
            with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'gt_density_map'), 'w') as hf:
                hf['gt_count'] = gt_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='ShanghaiTech')
    parser.add_argument('--input_dir', type=str, default='datasets/ShanghaiTech')
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    main(args)