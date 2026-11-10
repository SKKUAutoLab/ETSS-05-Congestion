import argparse
import os
from collections import defaultdict

def parse_frame_name(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    video_name, frame_id = stem.rsplit('_', 1)
    return video_name, int(frame_id), stem

def collect_images(image_dir):
    groups = defaultdict(list)
    for name in os.listdir(image_dir):
        if not name.lower().endswith('.jpg'):
            continue
        path = os.path.join(image_dir, name)
        video_name, frame_id, stem = parse_frame_name(path)
        groups[video_name].append((frame_id, stem, path))
    for video_name in groups:
        groups[video_name].sort(key=lambda item: item[0])
    return dict(sorted(groups.items()))

def allocate_balanced_counts(groups, total_target):
    video_names = list(groups.keys())
    base = total_target // len(video_names)
    remainder = total_target % len(video_names)
    allocations = {}
    for idx, video_name in enumerate(video_names):
        target = base + (1 if idx < remainder else 0)
        available = len(groups[video_name])
        allocations[video_name] = min(target, available)
    leftover = total_target - sum(allocations.values())
    while leftover > 0:
        candidates = [video_name for video_name in video_names if allocations[video_name] < len(groups[video_name])]
        if not candidates:
            raise RuntimeError(f'Cannot allocate {total_target} samples; only {sum(len(v) for v in groups.values())} frames are available.')
        for video_name in candidates:
            if leftover == 0:
                break
            allocations[video_name] += 1
            leftover -= 1
    return allocations

def evenly_select(items, target_count):
    if target_count >= len(items):
        return set(stem for _, stem, _ in items)
    if target_count <= 0:
        return set()
    selected = set()
    last_index = len(items) - 1
    last_target = target_count - 1
    for idx in range(target_count):
        source_index = round(idx * last_index / last_target) if last_target > 0 else 0
        selected.add(items[source_index][1])
    return selected

def prune_split(split_dir, target_count, apply=False):
    image_dir = os.path.join(split_dir, 'images')
    txt_dir = os.path.join(split_dir, 'txt')
    groups = collect_images(image_dir)
    allocations = allocate_balanced_counts(groups, target_count)
    keep_stems = set()
    for video_name, items in groups.items():
        keep_stems.update(evenly_select(items, allocations[video_name]))
    all_stems = {stem for items in groups.values() for _, stem, _ in items}
    delete_stems = all_stems - keep_stems
    print(f'\n{split_dir}')
    print(f'original: {len(all_stems)}')
    print(f'keep: {len(keep_stems)}')
    print(f'delete: {len(delete_stems)}')
    for video_name, items in groups.items():
        kept = sum(1 for _, stem, _ in items if stem in keep_stems)
        interval = len(items) / kept if kept else 0
        print(f'{video_name}: {len(items)} -> {kept} (interval ~{interval:.2f})')
    if not apply:
        print('dry run only; pass --apply to delete files')
        return
    removed_images = 0
    removed_txt = 0
    for stem in sorted(delete_stems):
        image_path = os.path.join(image_dir, stem + '.jpg')
        txt_path = os.path.join(txt_dir, stem + '.txt')
        if os.path.exists(image_path):
            os.remove(image_path)
            removed_images += 1
        if os.path.exists(txt_path):
            os.remove(txt_path)
            removed_txt += 1
    print(f'removed images: {removed_images}')
    print(f'removed txt: {removed_txt}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/Suwon')
    parser.add_argument('--train_target', type=int, default=10000)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()

    train_dir = os.path.join(args.input_dir, 'train_data')
    test_dir = os.path.join(args.input_dir, 'test_data')
    train_groups = collect_images(os.path.join(train_dir, 'images'))
    test_groups = collect_images(os.path.join(test_dir, 'images'))
    train_total = sum(len(v) for v in train_groups.values())
    test_total = sum(len(v) for v in test_groups.values())
    ratio = args.train_target / train_total
    test_target = round(test_total * ratio)
    print(f'train total: {train_total}')
    print(f'test total: {test_total}')
    print(f'ratio: {ratio:.6f}')
    print(f'train target: {args.train_target}')
    print(f'test target: {test_target}')
    prune_split(train_dir, args.train_target, apply=args.apply)
    prune_split(test_dir, test_target, apply=args.apply)

if __name__ == '__main__':
    main()