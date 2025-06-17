import os
import cv2
import h5py
import numpy as np
import scipy.io
import scipy.spatial
from scipy.ndimage.filters import gaussian_filter
import glob
import argparse
import warnings
warnings.filterwarnings("ignore")

def main(args):
    img_train_path = os.path.join(args.input_dir, 'Train')
    img_test_path = os.path.join(args.input_dir, 'Test')
    path_sets = [img_train_path, img_test_path]
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
    img_paths.sort()
    for img_path in img_paths:
        Img_data = cv2.imread(img_path) # [1875, 2500, 3]
        Gt_data = scipy.io.loadmat(img_path.replace('.jpg', '_ann.mat'))
        Gt_data = Gt_data['annPoints'] # [977, 2]
        if Img_data.shape[1] >= Img_data.shape[0]:
            rate_1 = 1152.0 / Img_data.shape[1]
            rate_2 = 768 / Img_data.shape[0]
            Img_data = cv2.resize(Img_data, (0, 0), fx=rate_1, fy=rate_2) # [768, 1152, 3]
            Gt_data[:, 0] = Gt_data[:, 0] * rate_1
            Gt_data[:, 1] = Gt_data[:, 1] * rate_2
        elif Img_data.shape[0] > Img_data.shape[1]:
            rate_1 = 1152.0 / Img_data.shape[0]
            rate_2 = 768.0 / Img_data.shape[1]
            Img_data = cv2.resize(Img_data, (0, 0), fx=rate_2, fy=rate_1) # [1152, 768, 3]
            Gt_data[:, 0] = Gt_data[:, 0] * rate_2
            Gt_data[:, 1] = Gt_data[:, 1] * rate_1
        kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))
        for count in range(0, len(Gt_data)):
            if int(Gt_data[count][1]) < Img_data.shape[0] and int(Gt_data[count][0]) < Img_data.shape[1]:
                kpoint[int(Gt_data[count][1]), int(Gt_data[count][0])] = 1
        height, width = Img_data.shape[0], Img_data.shape[1]
        m = int(width / 384)
        n = int(height / 384)
        fname = img_path.split('/')[-1]
        root_path = img_path.split('img_')[0]
        if not os.path.exists(root_path.replace('Train', 'train_data/images')):
            os.makedirs(root_path.replace('Train', 'train_data/images'))
        if not os.path.exists(root_path.replace('Train', 'train_data/gt_density_map')):
            os.makedirs(root_path.replace('Train', 'train_data/gt_density_map'))
        if not os.path.exists(root_path.replace('Test', 'test_data/images')):
            os.makedirs(root_path.replace('Test', 'test_data/images'))
        if not os.path.exists(root_path.replace('Test', 'test_data/gt_density_map')):
            os.makedirs(root_path.replace('Test', 'test_data/gt_density_map'))
        if root_path.split('/')[-2] == 'Train':
            for i in range(0, m):
                for j in range(0, n):
                    crop_img = Img_data[j * 384: 384 * (j + 1), i * 384:(i + 1) * 384] # [384, 384, 3]
                    crop_kpoint = kpoint[j * 384: 384 * (j + 1), i * 384:(i + 1) * 384] # [384, 384]
                    gt_count = np.sum(crop_kpoint)
                    save_fname = str(i) + str(j) + str('_') + fname
                    save_path = root_path.replace('Train', 'train_data/images') + save_fname
                    h5_path = save_path.replace('.jpg', '.h5').replace('images', 'gt_density_map')
                    with h5py.File(h5_path, 'w') as hf:
                        hf['gt_count'] = gt_count
                    cv2.imwrite(save_path, crop_img)
                    density_map = gaussian_filter(crop_kpoint, 2) # [384, 384]
                    density_map = density_map / np.max(density_map) * 255 # [384, 384]
                    density_map = density_map.astype(np.uint8) # [384, 384]
                    density_map = cv2.applyColorMap(density_map, 2) # [384, 384, 3]
                    result = np.hstack((density_map, crop_img)) # [384, 768, 3]
                    cv2.imwrite(save_path.replace('images', 'gt_show').replace('jpg', 'jpg'), result)
        else:
            save_path = root_path.replace('Test', 'test_data/images') + fname
            cv2.imwrite(save_path, Img_data)
            gt_count = np.sum(kpoint)
            with h5py.File(save_path.replace('.jpg', '.h5').replace('images', 'gt_density_map'), 'w') as hf:
                hf['gt_count'] = gt_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='qnrf')
    parser.add_argument('--input_dir', type=str, default='datasets/UCF-QNRF_ECCV18')
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    main(args)