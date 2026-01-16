import os
import shutil
import numpy as np
from PIL import Image
import cv2
import scipy.io as io
import json
from scipy.io import savemat
import argparse

def process_ucf_qnrf(data_root, down_size=1536):
    def load_data(img_gt_path, down_size):
        img_path, gt_path = img_gt_path
        img = cv2.imread(img_path)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        anno = io.loadmat(gt_path)
        points = anno["annPoints"]
        maxH = maxW = down_size
        img_w, img_h = img.size
        fw, fh = img_w / maxW, img_h / maxH
        maxf = max(fw, fh)
        factor = maxf if maxf > 1.0 else 1.0
        img = img.resize((int(img_w / factor), int(img_h / factor)))
        points = points / factor
        return img, points, anno
    
    img_quality = 100
    new_data_root = f"data/UCF-QNRF_{down_size}"
    os.makedirs(new_data_root, exist_ok=True)
    splits = ["Train", "Test"]
    for split in splits:
        img_list = os.listdir(f"{data_root}/{split}")
        img_list = [img for img in img_list if ".jpg" in img]
        gt_list = {}
        for img_name in img_list:
            img_path = f"{data_root}/{split}/{img_name}"
            gt_list[img_path] = img_path.replace(".jpg", "_ann.mat")
        img_list = sorted(list(gt_list.keys()))
        for img_path in img_list:
            gt_path = gt_list[img_path]
            img, points, anno = load_data((img_path, gt_path), down_size)
            new_img_path = img_path.replace(data_root, new_data_root)
            new_gt_path = gt_list[img_path].replace(data_root, new_data_root)
            save_dir = '/'.join(new_img_path.split('/')[:-1])
            os.makedirs(save_dir, exist_ok=True)
            img.save(new_img_path, quality=img_quality)
            anno["annPoints"] = points
            savemat(new_gt_path, anno)
            print("Save image to:", new_img_path)

def process_jhu_crowd(data_root, down_size=2048):
    def load_data(img_gt_path, down_size):
        img_path, gt_path = img_gt_path
        img = cv2.imread(img_path)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        file = open(gt_path, 'r')
        lines = file.readlines()
        if len(lines) > 0:
            points = np.array([line.split(' ')[:2] for line in lines]).astype(float)
            points = points
        else:
            points = np.empty((0,2))
        img_w, img_h = img.size
        max_h = max_w = down_size
        fw, fh = img_w / max_w, img_h / max_h
        maxf = max(fw, fh)
        factor = maxf if maxf > 1.0 else 1.0
        img = img.resize((int(img_w / factor),int(img_h / factor)))
        points = points / factor
        return img, points, lines

    img_quality = 100
    splits = ["train", "val", "test"]
    for split in splits:
        img_list = os.listdir(f"{data_root}/{split}/images")
        img_list = [img for img in img_list if ".jpg" in img]
        gt_list = {}
        for img_name in img_list:
            img_path = f"{data_root}/{split}/images/{img_name}"
            gt_list[img_path] = f"{data_root}/{split}/gt/{img_name}".replace("jpg", "txt")
        img_list = sorted(list(gt_list.keys()))
        new_data_root = f"data/JHU_Crowd_{down_size}"
        os.makedirs(f"{new_data_root}/{split}/images", exist_ok=True)
        os.makedirs(f"{new_data_root}/{split}/gt", exist_ok=True)
        img_label = f"{data_root}/{split}/image_labels.txt"
        new_img_label = f"{new_data_root}/{split}/image_labels.txt"
        shutil.copyfile(img_label, new_img_label)
        for index in range(len(img_list)):
            img_path = img_list[index]
            gt_path = gt_list[img_path]
            img, points, lines = load_data((img_path, gt_path), down_size)
            new_img_path = img_path.replace(data_root, new_data_root)
            new_gt_path = gt_list[img_path].replace(data_root, new_data_root)
            img.save(new_img_path, quality=img_quality)
            with open(new_gt_path, 'w') as file:
                for line, point in zip(lines, points):
                    line_split = line.split(' ')
                    new_point = [str(int(i)) for i in point.tolist()]
                    new_line = new_point + line_split[2:]
                    new_line = ' '.join(new_line)
                    file.write(new_line)
            print("Save image to:", new_img_path)

def process_nwpu_crowd(data_root, down_size=2048):
    def load_data(img_gt_path, down_size):
        img_path, gt_path = img_gt_path
        img = cv2.imread(img_path)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))    
        anno = json.load(open(gt_path, 'r'))
        points = np.array(anno['points'])
        if len(points) == 0:
            points = np.empty((0, 2))
        img_w, img_h = img.size
        max_h = max_w = down_size
        fw, fh = img_w / max_w, img_h / max_h
        maxf = max(fw, fh)
        factor = maxf if maxf > 1.0 else 1.0
        img = img.resize((int(img_w / factor),int(img_h / factor)))
        points = points / factor
        return img, points, anno
    
    def load_data_test(img_gt_path, down_size):
        img_path = img_gt_path
        img = cv2.imread(img_path)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_w, img_h = img.size
        max_h = max_w = down_size
        fw, fh = img_w / max_w, img_h / max_h
        maxf = max(fw, fh)
        factor = maxf if maxf > 1.0 else 1.0
        img = img.resize((int(img_w / factor),int(img_h / factor)))
        return img
    
    img_quality = 100
    splits = ["train", "val", "test"]
    for split in splits:
        list_name = f"{data_root}/{split}.txt"
        list_file = open(list_name, 'r')
        img_list = list_file.readlines()
        img_list = [img.split(' ')[0] for img in img_list]
        gt_list = {}
        for img_name in img_list:
            img_path = f"{data_root}/images/{img_name}.jpg"
            dotmap_path = f"{data_root}/jsons/{img_name}.json"
            gt_list[img_path] = dotmap_path
        img_list = sorted(list(gt_list.keys()))
        new_data_root = f"data/NWPU-Crowd_{down_size}"
        os.makedirs(f"{new_data_root}/images", exist_ok=True)
        os.makedirs(f"{new_data_root}/jsons", exist_ok=True)
        shutil.copyfile(list_name, f"{new_data_root}/{split}.txt")
        for img_path in img_list:
            gt_path = gt_list[img_path]
            if split == 'test':
                img = load_data_test(img_path, down_size)
            else:
                img, points, anno = load_data((img_path, gt_path), down_size)
            new_img_path = img_path.replace(data_root, new_data_root)
            new_gt_path = gt_list[img_path].replace(data_root, new_data_root)
            img.save(new_img_path, quality=img_quality)
            if split != 'test':
                anno['points'] = points.tolist()
                with open(new_gt_path, 'w') as f:
                    json.dump(anno, f)
            print("Save image to:", new_img_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='qnrf')
    parser.add_argument('--input_dir', type=str, default='data/UCF-QNRF')
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    if args.type_dataset == "qnrf":
        down_size = 1536
        process_ucf_qnrf(args.input_dir, down_size=down_size)
    elif args.type_dataset == "jhu":
        down_size = 2048
        process_jhu_crowd(args.input_dir, down_size=down_size)
    elif args.type_dataset == "nwpu":
        down_size = 2048
        process_nwpu_crowd(args.input_dir, down_size=down_size)
    else:
        print('This dataset does not exist')
        raise NotImplementedError