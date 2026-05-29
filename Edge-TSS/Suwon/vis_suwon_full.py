import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

json_path = 'train_suwon.json'
images_folder = 'processed_images'
output_folder = 'vis_suwon'
os.makedirs(output_folder, exist_ok=True)
with open(json_path, 'r') as f:
    data = json.load(f)
images = data['images']
annotations = data['annotations']
categories = data['categories']
cat_id_to_name = {cat['id']: cat['name'] for cat in categories}
full_class_names = ['unidentified', 'others', 'pedestrian', 'micromobility', 'car', 'bus', 'small truck', 'truck', 'mask']
color_list = ['gray', 'brown', 'lime', 'cyan', 'red', 'orange', 'magenta', 'yellow', 'blue']
colors = {}
for cat in categories:
    cat_id = cat['id']
    name = cat['name']
    if name in full_class_names:
        index = full_class_names.index(name)
        colors[cat_id] = color_list[index]
    else:
        colors[cat_id] = 'white'
ann_dict = {}
for ann in annotations:
    img_id = ann['image_id']
    ann_dict.setdefault(img_id, []).append(ann)
print(f"Visualizing {len(images)} images with full 8 classes + mask...")
print("Classes:", " | ".join(full_class_names))
print()
processed = 0
for img_info in images:
    img_id = img_info['id']
    file_name = img_info['file_name']
    img_path = os.path.join(images_folder, file_name)
    if not os.path.exists(img_path):
        print(f"Warning: Image not found → {img_path}")
        continue
    image = Image.open(img_path).convert("RGB")
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    ax.axis('off')
    if img_id in ann_dict:
        for ann in ann_dict[img_id]:
            bbox = ann['bbox']
            category_id_0based = ann['category_id']
            category_id_1based = category_id_0based + 1
            class_name = cat_id_to_name.get(category_id_1based, f'unknown_{category_id_0based}')
            color = colors.get(category_id_1based, 'white')
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2.5, edgecolor=color, facecolor='none', linestyle='-')
            ax.add_patch(rect)
            ax.text(bbox[0], bbox[1] - 6, class_name, color='white', fontsize=9, fontweight='bold', bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=2.5))
    output_path = os.path.join(output_folder, file_name)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close(fig)
    processed += 1
    if processed % 500 == 0:
        print(f"Processed {processed}/{len(images)} images...")
print("\nVisualization complete!")
print(f"All visualized images saved to: '{output_folder}'")
print("Now showing all classes: unidentified, others, pedestrian, micromobility, car, bus, small truck, truck, mask")
