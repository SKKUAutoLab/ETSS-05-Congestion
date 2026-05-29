import os
import shutil
from collections import defaultdict

images_dir = 'images'
txt_dir = 'txt'
proc_img_dir = 'processed_images'
proc_txt_dir = 'processed_txt'
total_needed = 5000
os.makedirs(proc_img_dir, exist_ok=True)
os.makedirs(proc_txt_dir, exist_ok=True)
img_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
if not img_files:
    print("No images found.")
    exit()
groups = defaultdict(list)
for f in img_files:
    base = os.path.splitext(f)[0]
    if '_' in base:
        prefix = base.rsplit('_', 1)[0]
        try:
            frame_idx = int(base.rsplit('_', 1)[1])
        except ValueError:
            frame_idx = 0
        groups[prefix].append((frame_idx, f))
prefixes = list(groups.keys())
prefixes.sort()
num_prefixes = len(prefixes)
print(f"Found {num_prefixes} unique videos.")
if num_prefixes == 0:
    print("No prefixes found.")
    exit()
for prefix in prefixes:
    groups[prefix].sort(key=lambda x: x[0])
    groups[prefix] = [filename for _, filename in groups[prefix]]
avail = {p: len(groups[p]) for p in prefixes}
selected_per_prefix = {p: 0 for p in prefixes}
remaining_needed = total_needed
base_per = remaining_needed // num_prefixes
for p in prefixes:
    take = min(base_per, avail[p])
    selected_per_prefix[p] = take
    remaining_needed -= take
i = 0
while remaining_needed > 0:
    p = prefixes[i % num_prefixes]
    if selected_per_prefix[p] < avail[p]:
        selected_per_prefix[p] += 1
        remaining_needed -= 1
    i += 1
total_avail = sum(avail.values())
if remaining_needed > 0:
    print(f"Balanced allocation reached {total_needed - remaining_needed}. Taking all available from remaining videos to maximize.")
    for p in prefixes:
        if selected_per_prefix[p] < avail[p]:
            selected_per_prefix[p] = avail[p]
copied = 0
for p in prefixes:
    num = selected_per_prefix[p]
    if num == 0:
        continue
    img_list = groups[p]
    avail_here = len(img_list)
    if num >= avail_here:
        selected_imgs = img_list
    else:
        step = avail_here / num
        indices = [int(round(j * step)) for j in range(num)]
        indices = sorted(set(indices))
        selected_imgs = [img_list[idx] for idx in indices[:num]]
    for sel_img in selected_imgs:
        shutil.copy2(os.path.join(images_dir, sel_img), os.path.join(proc_img_dir, sel_img))
        base_name = os.path.splitext(sel_img)[0]
        txt_name = base_name + '.txt'
        src_txt = os.path.join(txt_dir, txt_name)
        if os.path.exists(src_txt):
            shutil.copy2(src_txt, os.path.join(proc_txt_dir, txt_name))
        else:
            print(f"Warning: Missing TXT file for {sel_img}")
    copied += len(selected_imgs)
print(f"Copied {copied} images and corresponding TXT files.")
print(f"Total available in source: {total_avail}")
