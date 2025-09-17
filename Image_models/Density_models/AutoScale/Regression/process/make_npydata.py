import os
import numpy as np
import argparse

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # process sha dataset
    try:
        shanghaiAtrain_path = os.path.join(args.sh_dir, 'part_A_final/train_data/images/')
        shanghaiAtest_path = os.path.join(args.sh_dir, 'part_A_final/test_data/images/')
        train_list = []
        for filename in os.listdir(shanghaiAtrain_path):
            if filename.split('.')[1] == 'jpg':
                train_list.append(shanghaiAtrain_path+filename)
        train_list.sort()
        np.save(os.path.join(args.output_dir, 'ShanghaiA_train.npy'), train_list)
        test_list = []
        for filename in os.listdir(shanghaiAtest_path):
            if filename.split('.')[1] == 'jpg':
                test_list.append(shanghaiAtest_path+filename)
        test_list.sort()
        np.save(os.path.join(args.output_dir, 'ShanghaiA_test.npy'), test_list)
    except:
        print("The ShanghaiA dataset path is wrong. Please check you path.")
    # process shb dataset
    try:
        shanghaiBtrain_path = os.path.join(args.sh_dir, 'part_B_final/train_data/images/')
        shanghaiBtest_path = os.path.join(args.sh_dir, 'part_B_final/test_data/images/')
        train_list = []
        for filename in os.listdir(shanghaiBtrain_path):
            if filename.split('.')[1] == 'jpg':
                train_list.append(shanghaiBtrain_path+filename)
        train_list.sort()
        np.save(os.path.join(args.output_dir, 'ShanghaiB_train.npy'), train_list)
        test_list = []
        for filename in os.listdir(shanghaiBtest_path):
            if filename.split('.')[1] == 'jpg':
                test_list.append(shanghaiBtest_path+filename)
        test_list.sort()
        np.save(os.path.join(args.output_dir, 'ShanghaiB_test.npy'), test_list)
    except:
        print("The ShanghaiB dataset path is wrong. Please check your path.")
    # process qnrf dataset
    try:
        Qnrf_train_path = os.path.join(args.qnrf_dir, 'train_data/images/')
        Qnrf_test_path = os.path.join(args.qnrf_dir, 'test_data/images/')
        train_list = []
        for filename in os.listdir(Qnrf_train_path):
            if filename.split('.')[1] == 'jpg':
                train_list.append(Qnrf_train_path+filename)
        train_list.sort()
        np.save(os.path.join(args.output_dir, 'qnrf_train.npy'), train_list)
        test_list = []
        for filename in os.listdir(Qnrf_test_path):
            if filename.split('.')[1] == 'jpg':
                test_list.append(Qnrf_test_path+filename)
        test_list.sort()
        np.save(os.path.join(args.output_dir, 'qnrf_test.npy'), test_list)
    except:
        print("The QNRF dataset path is wrong. Please check your path.")
    # process jhu dataset
    try:
        Jhu_train_path = os.path.join(args.jhu_dir, 'train/images_2048/')
        Jhu_val_path = os.path.join(args.jhu_dir, 'val/images_2048/')
        jhu_test_path = os.path.join(args.jhu_dir, 'test/images_2048/')
        train_list = []
        for filename in os.listdir(Jhu_train_path):
            if filename.split('.')[1] == 'jpg':
                train_list.append(Jhu_train_path+filename)
        train_list.sort()
        np.save(os.path.join(args.output_dir, 'jhu_train.npy'), train_list)
        val_list = []
        for filename in os.listdir(Jhu_val_path):
            if filename.split('.')[1] == 'jpg':
                val_list.append(Jhu_val_path + filename)
        val_list.sort()
        np.save(os.path.join(args.output_dir, 'jhu_val.npy'), val_list)
        test_list = []
        for filename in os.listdir(jhu_test_path):
            if filename.split('.')[1] == 'jpg':
                test_list.append(jhu_test_path+filename)
        test_list.sort()
        np.save(os.path.join(args.output_dir, 'jhu_test.npy'), test_list)
    except:
        print("The JHU dataset path is wrong. Please check your path.")
    # process nwpu dataset
    try:
        f = open(os.path.join(args.nwpu_dir, "train.txt"), "r")
        train_list = f.readlines()
        f = open(os.path.join(args.nwpu_dir, "val.txt"), "r")
        val_list = f.readlines()
        f = open(os.path.join(args.nwpu_dir, "test.txt"), "r")
        test_list = f.readlines()
        root = os.path.join(args.nwpu_dir, 'images_1024/')
        train_img_list = []
        for i in range(len(train_list)):
            fname = train_list[i].split(' ')[0] + '.jpg'
            train_img_list.append(root + fname)
        np.save(os.path.join(args.output_dir, 'nwpu_train_1024.npy'), train_img_list)
        val_img_list = []
        for i in range(len(val_list)):
            fname = val_list[i].split(' ')[0] + '.jpg'
            val_img_list.append(root + fname)
        np.save(os.path.join(args.output_dir, 'nwpu_val_1024.npy'), val_img_list)
        test_img_list = []
        root = root.replace('images','test_data')
        for i in range(len(test_list)):
            fname = test_list[i].split(' ')[0] + '.jpg'
            test_img_list.append(root + fname)
        np.save(os.path.join(args.output_dir, 'nwpu_test_1024.npy'), test_img_list)
    except:
        print("The NWPU dataset path is wrong. Please check your path.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sh_dir', type=str, default='datasets/ShanghaiTech')
    parser.add_argument('--qnrf_dir', type=str, default='datasets/UCF-QNRF')
    parser.add_argument('--jhu_dir', type=str, default='datasets/jhu_crowd_v2.0')
    parser.add_argument('--nwpu_dir', type=str, default='datasets/NWPU_regression')
    parser.add_argument('--output_dir', type=str, default='npydata')
    args = parser.parse_args()
    main(args)