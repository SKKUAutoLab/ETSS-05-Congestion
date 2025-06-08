import os
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jhu_path', type=str, default='data/jhu_crowd_v2.0')
    parser.add_argument('--nwpu_path', type=str, default='data/NWPU_CLTR')
    parser.add_argument('--output_dir', type=str, default='npydata')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    try:
        Jhu_train_path = args.jhu_path + '/train/images_2048/'
        Jhu_val_path = args.jhu_path + '/val/images_2048/'
        jhu_test_path = args.jhu_path + '/test/images_2048/'
        train_list = []
        for filename in os.listdir(Jhu_train_path):
            if filename.split('.')[1] == 'jpg':
                train_list.append(Jhu_train_path + filename)
        train_list.sort()
        np.save(args.output_dir + '/jhu_train.npy', train_list)
        val_list = []
        for filename in os.listdir(Jhu_val_path):
            if filename.split('.')[1] == 'jpg':
                val_list.append(Jhu_val_path + filename)
        val_list.sort()
        np.save(args.output_dir + '/jhu_val.npy', val_list)
        test_list = []
        for filename in os.listdir(jhu_test_path):
            if filename.split('.')[1] == 'jpg':
                test_list.append(jhu_test_path + filename)
        test_list.sort()
        np.save(args.output_dir + '/jhu_test.npy', test_list)
    except:
        print("The JHU dataset path is wrong.")
    try:
        f = open("data/NWPU_list/train.txt", "r")
        train_list = f.readlines()
        f = open("data/NWPU_list/val.txt", "r")
        val_list = f.readlines()
        root = args.nwpu_path + '/gt_detr_map/'
        if not os.path.exists(root):
            print("The NWPU dataset path is wrong.")
        else:
            train_img_list = []
            for i in range(len(train_list)):
                fname = train_list[i].split(' ')[0] + '.jpg'
                train_img_list.append(root + fname)
            val_img_list = []
            for i in range(len(val_list)):
                fname = val_list[i].split(' ')[0] + '.jpg'
                val_img_list.append(root + fname)
            np.save(args.output_dir + '/nwpu_train.npy', train_img_list)
            np.save(args.output_dir + '/nwpu_val.npy', val_img_list)
    except:
        print("The NWPU dataset path is wrong.")