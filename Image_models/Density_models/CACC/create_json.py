import json
import glob
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='datasets/ShanghaiTech/part_B')
    args = parser.parse_args()

    for mode in ['train_data', 'test_data']:
        output_json = os.path.join('datasets', mode.split('_')[0] + '_' + args.input_dir.split('/')[-1] + '.json')
        img_list = []
        for img_path in glob.glob(os.path.join(args.input_dir, mode, 'images/*.jpg')):
            img_list.append(img_path)
        with open(output_json, 'w') as f:
            json.dump(img_list, f)