# coding: utf-8

import argparse
import glob
import os

import cv2
import h5py
import numpy as np
import scipy.io as io
from PIL import Image

parser = argparse.ArgumentParser(description='IncepTNet')
parser.add_argument('--data_path', type=str, default='./datasets/ShanghaiTech',
                    help='the data path of ShanghaiTech')

args = parser.parse_args()
root = os.path.abspath(args.data_path)

part_A_train = os.path.join(root, 'part_A_final', 'train_data', 'images')
part_A_test = os.path.join(root, 'part_A_final', 'test_data', 'images')

'''mkdir directories'''
if not os.path.exists(part_A_train.replace('images', 'gt_detr_map')):
    os.makedirs(part_A_train.replace('images', 'gt_detr_map'))
if not os.path.exists(part_A_test.replace('images', 'gt_detr_map')):
    os.makedirs(part_A_test.replace('images', 'gt_detr_map'))

path_sets = [part_A_train, part_A_test]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

img_paths.sort()

for img_path in img_paths:

    img = cv2.imread(img_path)
    Img_data_pil = Image.open(img_path).convert('RGB')

    print(img_path)
    rate = 1
    rate1 = 1
    rate2 = 1
    if img.shape[1] >= img.shape[0] and img.shape[1] >= 2048:
        rate1 = 2048.0 / img.shape[1]
    elif img.shape[0] >= img.shape[1] and img.shape[0] >= 2048:
        rate1 = 2048.0 / img.shape[0]
    img = cv2.resize(img, (0, 0), fx=rate1, fy=rate1, interpolation=cv2.INTER_CUBIC)
    Img_data_pil = Img_data_pil.resize((img.shape[1], img.shape[0]), Image.LANCZOS)

    min_shape = 512.0
    if img.shape[1] <= img.shape[0] and img.shape[1] <= min_shape:
        rate2 = min_shape / img.shape[1]
    elif img.shape[0] <= img.shape[1] and img.shape[0] <= min_shape:
        rate2 = min_shape / img.shape[0]
    img = cv2.resize(img, (0, 0), fx=rate2, fy=rate2, interpolation=cv2.INTER_CUBIC)
    Img_data_pil = Img_data_pil.resize((img.shape[1], img.shape[0]), Image.LANCZOS)

    rate = rate1 * rate2

    k = np.zeros((img.shape[0], img.shape[1]))
    mat = io.loadmat(
        img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
    gt_file = mat["image_info"][0][0][0][0][0]

    y = gt_file[:, 0] * rate
    x = gt_file[:, 1] * rate
    for i in range(0, len(x)):
        if int(x[i]) < img.shape[0] and int(y[i]) < img.shape[1]:
            k[int(x[i]), int(y[i])] += 1

    kpoint = k.copy()
    kpoint = kpoint.astype(np.uint8)

    with h5py.File(img_path.replace('images', 'gt_detr_map').replace('jpg', 'h5'), 'w') as hf:
        hf['kpoint'] = kpoint
        hf['image'] = np.asarray(Img_data_pil)

'''generate the image list'''
if not os.path.exists('./npydata'):
    os.makedirs('./npydata')

train_list = []
for filename in os.listdir(part_A_train):
    if filename.split('.')[1] == 'jpg':
        train_list.append(os.path.join(part_A_train, filename))
train_list.sort()
np.save('./npydata/sha_train.npy', train_list)

test_list = []
for filename in os.listdir(part_A_test):
    if filename.split('.')[1] == 'jpg':
        test_list.append(os.path.join(part_A_test, filename))
test_list.sort()
np.save('./npydata/sha_test.npy', test_list)

print("Generate ShanghaiTech part_A image list successfully", len(train_list), len(test_list))
print("end")
