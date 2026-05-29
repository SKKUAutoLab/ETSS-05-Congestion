import os
import shutil

images_source_dir = 'images'
labels_source_dir = 'labels'
processed_images_dir = 'processed_images'
processed_labels_dir = 'processed_labels'
os.makedirs(processed_images_dir, exist_ok=True)
os.makedirs(processed_labels_dir, exist_ok=True)
print("Copying images...")
jpg_count = 0
for root, dirs, files in os.walk(images_source_dir):
    for file in files:
        if file.lower().endswith('.jpg'):
            src_path = os.path.join(root, file)
            dst_path = os.path.join(processed_images_dir, file)
            shutil.copy2(src_path, dst_path)
            jpg_count += 1
print(f"Copied {jpg_count} .jpg files to '{processed_images_dir}'")
print("Copying labels...")
txt_count = 0
for root, dirs, files in os.walk(labels_source_dir):
    for file in files:
        if file.lower().endswith('.txt'):
            src_path = os.path.join(root, file)
            dst_path = os.path.join(processed_labels_dir, file)
            shutil.copy2(src_path, dst_path)
            txt_count += 1
print(f"Copied {txt_count} .txt files to '{processed_labels_dir}'")
print("All done!")
