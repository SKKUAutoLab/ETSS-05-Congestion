import glob
import os

import h5py
import numpy as np
import scipy.io as io
from PIL import Image
from scipy.ndimage import gaussian_filter

ROOT = os.path.dirname(os.path.abspath(__file__))

# raw data (downloaded)
SHA_SRC = os.path.join(ROOT, 'data', 'ShanghaiTech', 'part_A', 'test_data')
JHU_SRC = os.path.join(ROOT, 'data', 'jhu_crowd_v2.0', 'test')

# processed data (expected by test_SHA.py / test_JHU.py)
SHA_DST = os.path.join(ROOT, 'Data_Crowd_Counting', 'ShanghaiTech_Crowd_Counting_Dataset',
                       'part_A_final', 'test_data')
JHU_DST = os.path.join(ROOT, 'Data_Crowd_Counting', 'JHU', 'test')


def link_image(src_path, dst_dir):
    dst_path = os.path.join(dst_dir, os.path.basename(src_path))
    if not os.path.exists(dst_path):
        os.symlink(src_path, dst_path)


def save_label(points, image_size, h5_path):
    width, height = image_size
    k = np.zeros((height, width), dtype=np.float32)
    count = 0
    for x, y in points:
        x, y = int(x), int(y)
        if 0 <= x < width and 0 <= y < height:
            k[y, x] = 1
            count += 1
    if len(points) - count > 0:
        print('Ignore {} wrong annotation.'.format(len(points) - count))
    k = gaussian_filter(k, 5)
    att = (k > 0.001).astype(np.float32)
    with h5py.File(h5_path, 'w') as hf:
        hf['density'] = k
        hf['attention'] = att
        hf['gt'] = count


def process_sha():
    image_dir = os.path.join(SHA_DST, 'images')
    label_dir = os.path.join(SHA_DST, 'new_data')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(SHA_SRC, 'images', '*.jpg')))
    for i, img_path in enumerate(img_paths):
        name = os.path.basename(img_path)
        print('[SHA {}/{}] {}'.format(i + 1, len(img_paths), name))
        mat_path = os.path.join(SHA_SRC, 'ground-truth',
                                'GT_' + name.replace('.jpg', '.mat'))
        mat = io.loadmat(mat_path)
        points = mat['image_info'][0, 0][0, 0][0]
        link_image(img_path, image_dir)
        save_label(points, Image.open(img_path).size,
                   os.path.join(label_dir, name.replace('.jpg', '.h5')))


def process_jhu():
    image_dir = os.path.join(JHU_DST, 'img')
    label_dir = os.path.join(JHU_DST, 'new_data_JHU')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(JHU_SRC, 'images', '*.jpg')))
    for i, img_path in enumerate(img_paths):
        name = os.path.basename(img_path)
        print('[JHU {}/{}] {}'.format(i + 1, len(img_paths), name))
        txt_path = os.path.join(JHU_SRC, 'gt', name.replace('.jpg', '.txt'))
        if os.path.getsize(txt_path) > 0:
            ann = np.loadtxt(txt_path, ndmin=2)
            points = ann[:, :2]
        else:
            points = np.zeros((0, 2), dtype=np.float32)
        link_image(img_path, image_dir)
        save_label(points, Image.open(img_path).size,
                   os.path.join(label_dir, name.replace('.jpg', '.h5')))


if __name__ == '__main__':
    process_sha()
    process_jhu()
