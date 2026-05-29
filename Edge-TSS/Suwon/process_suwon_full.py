import os
from PIL import Image
import json

images_folder = 'processed_images'
txt_folder = 'processed_labels'
output_json = 'train_suwon.json'
print("Starting YOLO to COCO conversion (fixed 8 classes + mask)...\n")
class_names = ['unidentified', 'others', 'pedestrian', 'micromobility', 'car', 'bus', 'small truck', 'truck']
yolo_to_coco_id = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8}
print("Class mapping:")
for yolo_id, coco_id in yolo_to_coco_id.items():
    print(f"YOLO class {yolo_id} → {class_names[coco_id - 1]} (COCO ID {coco_id})")
print()
print("Cleaning dataset...")
jpg_files = [f for f in os.listdir(images_folder) if f.lower().endswith('.jpg')]
removed = 0
for file_name in jpg_files[:]:
    txt_path = os.path.join(txt_folder, os.path.splitext(file_name)[0] + '.txt')
    if not os.path.exists(txt_path):
        os.remove(os.path.join(images_folder, file_name))
        print(f"Removed (no TXT): {file_name}")
        removed += 1
        continue
    with open(txt_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        os.remove(os.path.join(images_folder, file_name))
        os.remove(txt_path)
        print(f"Removed (empty TXT): {file_name}")
        removed += 1
print(f"Cleaning done. Removed {removed} images.\n")
categories = []
coco_1based_to_0based = {}
for cid, name in enumerate(class_names, start=1):
    categories.append({"supercategory": name, "id": cid, "name": name})
    coco_1based_to_0based[cid] = cid - 1
mask_id = 9
categories.append({"supercategory": "mask", "id": mask_id, "name": "mask"})
coco_1based_to_0based[mask_id] = mask_id - 1
print("Final 9 categories:")
for cat in categories:
    print(f"  ID {cat['id']}: {cat['name']}")
print()
print("Converting annotations...")
remaining_jpgs = sorted([f for f in os.listdir(images_folder) if f.lower().endswith('.jpg')])
images = []
annotations = []
ann_id = 1
img_id = 0
for file_name in remaining_jpgs:
    img_path = os.path.join(images_folder, file_name)
    with Image.open(img_path) as img:
        width, height = img.size
    images.append({"file_name": file_name, "height": height, "width": width, "id": img_id})
    txt_path = os.path.join(txt_folder, os.path.splitext(file_name)[0] + '.txt')
    if not os.path.exists(txt_path):
        img_id += 1
        continue
    with open(txt_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            yolo_class = int(parts[0])
        except:
            continue
        if yolo_class not in yolo_to_coco_id:
            print(f"Warning: Skipping unknown class {yolo_class} in {file_name}")
            continue
        cx, cy, nw, nh = map(float, parts[1:])
        x_min = round((cx - nw / 2) * width)
        y_min = round((cy - nh / 2) * height)
        w_box = round(nw * width)
        h_box = round(nh * height)
        coco_1based = yolo_to_coco_id[yolo_class]
        category_id_0based = coco_1based_to_0based[coco_1based]
        annotations.append({"category_id": category_id_0based, "bbox": [x_min, y_min, w_box, h_box], "area": w_box * h_box, "iscrowd": 0, "image_id": img_id, "id": ann_id, "ignore": 0})
        ann_id += 1
    img_id += 1
data = {"images": images, "annotations": annotations, "categories": categories}
with open(output_json, 'w') as f:
    json.dump(data, f, separators=(',', ':'))
print(f"\nConversion complete!")
print(f"Images: {len(images)}")
print(f"Annotations: {len(annotations)}")
print(f"Categories: {len(categories)} (8 classes + mask)")
print(f"Saved to: {output_json}")
