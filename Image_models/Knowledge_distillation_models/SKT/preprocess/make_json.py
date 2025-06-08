import json
import os
import argparse
import glob

def get_val(args):
    if args.type_dataset == 'sha':
        with open("preprocess/part_A_val.json") as f:
            val_list = json.load(f)
        new_val = []
        for item in val_list:
            new_item = item.replace('/home/leeyh/Downloads/Shanghai/part_A_final', args.input_dir)
            new_val.append(new_item)
        with open(args.val_json, 'w') as f:
            json.dump(new_val, f)
    elif args.type_dataset == 'shb':
        path = os.path.join(args.input_dir, 'train_data', 'images')
        filenames = os.listdir(path)
        pathname = [os.path.join(path, filename) for filename in filenames]
        with open(args.val_json, 'w') as f:
            json.dump(pathname, f)
    elif args.type_dataset == 'qnrf':
        filenames = glob.glob(os.path.join(args.input_dir, 'Train/*.jpg'))
        with open(args.val_json, 'w') as f:
            json.dump(filenames, f)
    else:
        print('This dataset does not exist')
        raise NotImplementedError

def get_train(args):
    if args.type_dataset == 'sha' or args.type_dataset == 'shb':
        path = os.path.join(args.input_dir, 'train_data', 'images')
        filenames = os.listdir(path)
        pathname = [os.path.join(path, filename) for filename in filenames]
        with open(args.train_json, 'w') as f:
            json.dump(pathname, f)
    elif args.type_dataset == 'qnrf':
        filenames = glob.glob(os.path.join(args.input_dir, 'Train/*.jpg'))
        with open(args.train_json, 'w') as f:
            json.dump(filenames, f)
    else:
        print('This dataset does not exist')
        raise NotImplementedError

def get_test(args):
    if args.type_dataset == 'sha' or args.type_dataset == 'shb':
        path = os.path.join(args.input_dir, 'test_data', 'images')
        filenames = os.listdir(path)
        pathname = [os.path.join(path, filename) for filename in filenames]
        with open(args.test_json, 'w') as f:
            json.dump(pathname, f)
    elif args.type_dataset == 'qnrf':
        filenames = glob.glob(os.path.join(args.input_dir, 'Test/*.jpg'))
        with open(args.test_json, 'w') as f:
            json.dump(filenames, f)
    else:
        print('This dataset does not exist')
        raise NotImplementedError

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='sha', choices=['sha', 'shb', 'qnrf'])
    parser.add_argument('--input_dir', type=str, default='datasets/ShanghaiTech/part_A_final')
    parser.add_argument('--train_json', type=str, default='A_train.json')
    parser.add_argument('--val_json', type=str, default='A_val.json')
    parser.add_argument('--test_json', type=str, default='A_test.json')
    args = parser.parse_args()

    get_train(args)
    get_val(args)
    get_test(args)