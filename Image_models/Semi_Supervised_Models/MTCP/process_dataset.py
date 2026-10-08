import os
import json
import glob
import numpy as np
import h5py
import scipy.io as io
from PIL import Image

from dataset.utils import get_density_map_gaussian


DATASETS = [
    {'data_name': 'SHA', 'src_root': 'dataset/ShanghaiTech/part_A', 'dst_root': 'dataset/SHA', 'label_rate': 0.2},
    {'data_name': 'SHB', 'src_root': 'dataset/ShanghaiTech/part_B', 'dst_root': 'dataset/SHB', 'label_rate': 0.05},
]


def load_points(gt_path):
    mat = io.loadmat(gt_path)
    points = mat['image_info'][0, 0][0, 0][0].astype(np.float32)
    return points


def process_split(src_root, dst_root, src_split, dst_split):
    src_img_dir = os.path.join(src_root, src_split, 'images')
    src_gt_dir = os.path.join(src_root, src_split, 'ground-truth')
    dst_img_dir = os.path.join(dst_root, dst_split, 'images')
    dst_h5_dir = os.path.join(dst_root, dst_split, 'h5pys')
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_h5_dir, exist_ok=True)

    img_files = sorted(glob.glob(os.path.join(src_img_dir, '*.jpg')),
                       key=lambda p: int(os.path.basename(p).replace('IMG_', '').replace('.jpg', '')))

    dst_img_paths = []
    for img_path in img_files:
        img_name = os.path.basename(img_path)
        gt_path = os.path.join(src_gt_dir, 'GT_' + img_name.replace('.jpg', '.mat'))
        points = load_points(gt_path)

        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        # resize so that both sides are divisible by 8 (density map is downsampled by 8 in the loader)
        new_w = max(8, w // 8 * 8)
        new_h = max(8, h // 8 * 8)
        ratio_w = new_w / w
        ratio_h = new_h / h
        if (new_w, new_h) != (w, h):
            img = img.resize((new_w, new_h), Image.BILINEAR)

        density = get_density_map_gaussian(new_h, new_w, ratio_h, ratio_w, points, fixed_value=15)

        dst_img_path = os.path.join(dst_img_dir, img_name)
        img.save(dst_img_path)
        with h5py.File(os.path.join(dst_h5_dir, img_name.replace('.jpg', '.h5')), 'w') as hf:
            hf['density'] = density

        dst_img_paths.append(dst_img_path)
        print(f'{dst_img_path}: gt {len(points)}, density sum {density.sum():.2f}')

    return dst_img_paths


def save_json(dst_root, path_list, json_name):
    json_path = os.path.join(dst_root, json_name)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(path_list, f, indent=4)
    print(f'{json_path}: {len(path_list)} images')


if __name__ == '__main__':
    for cfg in DATASETS:
        data_name = cfg['data_name']
        src_root = cfg['src_root']
        dst_root = cfg['dst_root']
        label_rate = cfg['label_rate']

        train_paths = process_split(src_root, dst_root, 'train_data', 'Train')
        test_paths = process_split(src_root, dst_root, 'test_data', 'Test')

        per_label = f'{int(label_rate * 100)}%'
        num_label = round(len(train_paths) * label_rate)

        save_json(dst_root, train_paths, data_name + '_train_all.json')
        save_json(dst_root, train_paths[:num_label], data_name + '_train_label_' + per_label + '.json')
        save_json(dst_root, train_paths[num_label:], data_name + '_train_unlabel_' + per_label + '.json')
        save_json(dst_root, test_paths, data_name + '_test.json')
