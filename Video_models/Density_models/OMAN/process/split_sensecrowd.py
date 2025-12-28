import os
import shutil
import random

if __name__ == '__main__':
    src_dir = "data/Sense/videos"
    test_dir = "data/Sense/test"
    train_dir = "data/Sense/train"
    val_dir = "data/Sense/val"
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    subfolders = [f for f in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, f))]
    for folder in subfolders:
        if folder.startswith("test_"):
            src_path = os.path.join(src_dir, folder)
            dst_path = os.path.join(test_dir, folder)
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    train_folders = [f for f in subfolders if f.startswith("train_")]
    random.shuffle(train_folders)
    split_idx = int(0.8 * len(train_folders))
    train_split = train_folders[:split_idx]
    val_split = train_folders[split_idx:]
    for folder in train_split:
        src_path = os.path.join(src_dir, folder)
        dst_path = os.path.join(train_dir, folder)
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    for folder in val_split:
        src_path = os.path.join(src_dir, folder)
        dst_path = os.path.join(val_dir, folder)
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    print("✅ Dataset reorganization completed!")