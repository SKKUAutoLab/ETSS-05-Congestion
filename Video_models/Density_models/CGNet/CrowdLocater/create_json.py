import os
import json
from PIL import Image
import argparse

def main(args):
    splits = {"train": {"txt": os.path.join(args.input_dir, "train.txt"), "img_dir": os.path.join(args.input_dir, "train"), "json": os.path.join(args.output_dir, 'train.json')},
              "val": {"txt": os.path.join(args.input_dir, "val.txt"), "img_dir": os.path.join(args.input_dir, "val"), "json": os.path.join(args.output_dir, 'val.json')},
              "test": {"txt": os.path.join(args.input_dir, "test.txt"), "img_dir": os.path.join(args.input_dir, "test"), "json": os.path.join(args.output_dir, 'test.json')}}
    for split, paths in splits.items():
        txt_file = paths["txt"]
        img_dir = paths["img_dir"]
        json_file = paths["json"]
        images_info = []
        img_id = 1
        with open(txt_file, "r") as f:
            if args.type_dataset == 'SENSE':
                subfolders = [line.strip() for line in f if line.strip()]
            elif args.type_dataset == 'HT21':
                subfolders = [line.strip().replace('train/', '').replace('val/', '').replace('test/', '') + '/img1' for line in f if line.strip()]
            else:
                print('This dataset does not exist')
                raise NotImplementedError
        for subfolder in subfolders:
            subfolder_path = os.path.join(img_dir, subfolder)
            if not os.path.isdir(subfolder_path):
                print(f"Warning: {subfolder_path} not found, skipping")
                continue
            for fname in sorted(os.listdir(subfolder_path)):
                fpath = os.path.join(subfolder_path, fname)
                if not os.path.isfile(fpath):
                    continue
                try:
                    with Image.open(fpath) as img:
                        width, height = img.size
                except Exception as e:
                    print(f"Error reading {fpath}: {e}")
                    continue
                images_info.append({"file_name": f"{subfolder}/{fname}", "height": height, "width": width, "id": img_id})
                img_id += 1
        output = {"images": images_info}
        with open(json_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved {json_file} with {len(images_info)} entries.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='SENSE', choices=['SENSE', 'HT21'])
    parser.add_argument('--input_dir', type=str, default='../datasets/Sense')
    parser.add_argument('--output_dir', type=str, default='datasets/SENSE')
    args = parser.parse_args()

    print('Create json for dataset:', args.type_dataset)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    main(args)