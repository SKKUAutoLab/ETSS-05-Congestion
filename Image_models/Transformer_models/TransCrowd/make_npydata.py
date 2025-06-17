import os
import numpy as np
import argparse

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.type_dataset == 'sha':
        try:
            shanghaiAtrain_path = os.path.join(args.input_dir, 'part_A_final/train_data/images_crop/')
            shanghaiAtest_path = os.path.join(args.input_dir, 'part_A_final/test_data/images_crop/')
            train_list = []
            for filename in os.listdir(shanghaiAtrain_path):
                if filename.split('.')[1] == 'jpg':
                    train_list.append(shanghaiAtrain_path + filename)
            train_list.sort()
            np.save(os.path.join(args.output_dir, 'ShanghaiA_train.npy'), train_list)
            test_list = []
            for filename in os.listdir(shanghaiAtest_path):
                if filename.split('.')[1] == 'jpg':
                    test_list.append(shanghaiAtest_path + filename)
            test_list.sort()
            np.save(os.path.join(args.output_dir, 'ShanghaiA_test.npy'), test_list)
            print("Generated sha image list:", len(train_list), len(test_list))
        except:
            print("The sha dataset path is wrong")
            raise NotImplementedError
    elif args.type_dataset == 'shb':
        try:
            shanghaiBtrain_path = os.path.join(args.input_dir, 'part_B_final/train_data/images_crop/')
            shanghaiBtest_path = os.path.join(args.input_dir, 'part_B_final/test_data/images_crop/')
            train_list = []
            for filename in os.listdir(shanghaiBtrain_path):
                if filename.split('.')[1] == 'jpg':
                    train_list.append(shanghaiBtrain_path + filename)
            train_list.sort()
            np.save(os.path.join(args.output_dir, 'ShanghaiB_train.npy'), train_list)
            test_list = []
            for filename in os.listdir(shanghaiBtest_path):
                if filename.split('.')[1] == 'jpg':
                    test_list.append(shanghaiBtest_path + filename)
            test_list.sort()
            np.save(os.path.join(args.output_dir, 'ShanghaiB_test.npy'), test_list)
            print("Generated shb image list", len(train_list), len(test_list))
        except:
            print("The shb dataset path is wrong")
            raise NotImplementedError
    elif args.type_dataset == 'qnrf':
        try:
            qnrftrain_path = os.path.join(args.input_dir, 'train_data/images/')
            qnrftest_path = os.path.join(args.input_dir, 'test_data/images/')
            train_list = []
            for filename in os.listdir(qnrftrain_path):
                if filename.split('.')[1] == 'jpg':
                    train_list.append(qnrftrain_path + filename)
            train_list.sort()
            np.save(os.path.join(args.output_dir, 'qnrf_train.npy'), train_list)
            test_list = []
            for filename in os.listdir(qnrftest_path):
                if filename.split('.')[1] == 'jpg':
                    test_list.append(qnrftest_path + filename)
            test_list.sort()
            np.save(os.path.join(args.output_dir, 'qnrf_test.npy'), test_list)
            print("Generated qnrf image list", len(train_list), len(test_list))
        except:
            print("The qnrf dataset path is wrong")
            raise NotImplementedError
    else:
        print('This dataset does not exist')
        raise NotImplementedError

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='sha', choices=['sha', 'shb', 'qnrf'])
    parser.add_argument('--input_dir', type=str, default='datasets/ShanghaiTech')
    parser.add_argument('--output_dir', type=str, default='npydata')
    args = parser.parse_args()
    main(args)