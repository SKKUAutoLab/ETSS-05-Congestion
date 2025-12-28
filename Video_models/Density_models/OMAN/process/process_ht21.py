import os
from PIL import Image
import glob
import argparse

def main(base_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    for subfolder in subfolders:
        subfolder_path = os.path.join(base_path, subfolder)
        img1_path = os.path.join(subfolder_path, "img1")
        if not os.path.exists(img1_path):
            print(f"Warning: img1 folder not found in {subfolder_path}")
            continue
        jpg_files = glob.glob(os.path.join(img1_path, "*.jpg"))
        if not jpg_files:
            print(f"Warning: No .jpg files found in {img1_path}")
            continue
        jpg_files.sort()
        output_file = os.path.join(output_path, f"{subfolder}.txt")
        print(f"Processing {subfolder}...")
        with open(output_file, 'w') as f:
            for jpg_file in jpg_files:
                try:
                    with Image.open(jpg_file) as img:
                        width, height = img.size
                    filename = os.path.join('img1', os.path.basename(jpg_file))
                    f.write(f"{filename} {width} {height}\n")
                except Exception as e:
                    print(f"Error processing {jpg_file}: {e}")
                    continue
        print(f"Created {output_file} with {len(jpg_files)} entries")
    print(f"\nProcessing complete! Label files created in: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='HT21')
    parser.add_argument('--input_dir', type=str, default='data/HT21/test')
    parser.add_argument('--output_dir', type=str, default='data/HT21/label_list_all')
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    main(args.input_dir, args.output_dir)