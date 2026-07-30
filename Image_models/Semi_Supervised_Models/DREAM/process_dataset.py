#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit("Missing dependency: pillow. Install it before running this script.") from exc

try:
    from scipy.io import loadmat
    from scipy.ndimage import gaussian_filter
except ImportError as exc:
    raise SystemExit("Missing dependency: scipy. Install it before running this script.") from exc


PARTS = ("part_A", "part_B")
SPLITS = ("train_data", "test_data")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process ShanghaiTech .mat annotations into density maps and file lists."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/ShanghaiTech"),
        help="ShanghaiTech dataset root.",
    )
    parser.add_argument(
        "--list-dir",
        type=Path,
        default=None,
        help="Directory for output file lists. Defaults to DATA_ROOT/file_lists.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=15.0,
        help="Gaussian sigma used to build density maps.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing *_densitymap.npy files.",
    )
    parser.add_argument(
        "--matlab-indexed",
        action="store_true",
        help="Subtract 1 from .mat point coordinates before placing them.",
    )
    parser.add_argument(
        "--train-min-size",
        type=int,
        default=256,
        help="Minimum width and height for images used by random training crops.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=None,
        help="Directory for padded training images. Defaults to DATA_ROOT/processed_train.",
    )
    return parser.parse_args()


def display_path(path, root):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def image_sort_key(path):
    stem = path.stem
    number = None
    if "_" in stem:
        tail = stem.rsplit("_", 1)[-1]
        if tail.isdigit():
            number = int(tail)
    return (stem.rsplit("_", 1)[0], number if number is not None else -1, stem)


def iter_point_arrays(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if not str(key).startswith("__"):
                yield from iter_point_arrays(item)
        return

    if isinstance(value, np.void):
        if value.dtype.names:
            for name in value.dtype.names:
                yield from iter_point_arrays(value[name])
        return

    if isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_point_arrays(item)
        return

    if not isinstance(value, np.ndarray):
        return

    if value.dtype == object:
        for item in value.flat:
            yield from iter_point_arrays(item)
        return

    if value.dtype.names:
        for name in value.dtype.names:
            yield from iter_point_arrays(value[name])
        return

    if not np.issubdtype(value.dtype, np.number):
        return

    arr = np.asarray(value)
    if arr.ndim != 2:
        return

    if arr.shape[1] >= 2:
        yield arr[:, :2]
    elif arr.shape[0] == 2:
        yield arr.T


def load_points(mat_path):
    mat = loadmat(str(mat_path))
    source = mat["image_info"] if "image_info" in mat else mat
    candidates = []

    for candidate in iter_point_arrays(source):
        arr = np.asarray(candidate, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue

        arr = arr[:, :2]
        arr = arr[np.isfinite(arr).all(axis=1)]
        if arr.shape[0] > 0:
            candidates.append(arr)

    if not candidates:
        return np.empty((0, 2), dtype=np.float32)

    return max(candidates, key=lambda item: item.shape[0]).astype(np.float32, copy=False)


def make_density(image_path, mat_path, density_path, sigma, matlab_indexed):
    with Image.open(image_path) as image:
        width, height = image.size

    points = load_points(mat_path)
    dots = np.zeros((height, width), dtype=np.float32)
    dropped = 0

    for point in points:
        x = float(point[0])
        y = float(point[1])
        if matlab_indexed:
            x -= 1.0
            y -= 1.0

        col = int(round(x))
        row = int(round(y))

        if 0 <= row < height and 0 <= col < width:
            dots[row, col] += 1.0
        else:
            dropped += 1

    if sigma > 0:
        density = gaussian_filter(dots, sigma=sigma, mode="constant")
        original_sum = float(dots.sum())
        density_sum = float(density.sum())
        if original_sum > 0.0 and density_sum > 0.0:
            density *= original_sum / density_sum
    else:
        density = dots

    np.save(str(density_path), density.astype(np.float32, copy=False))
    return int(dots.sum()), dropped


def make_train_ready_image(
    image_path,
    density_path,
    processed_image_path,
    processed_density_path,
    min_size,
    overwrite,
):
    with Image.open(image_path) as image:
        width, height = image.size

        if width >= min_size and height >= min_size:
            return image_path, density_path, False

        new_width = max(width, min_size)
        new_height = max(height, min_size)

        if processed_image_path.exists() and processed_density_path.exists() and not overwrite:
            return processed_image_path, processed_density_path, True

        processed_image_path.parent.mkdir(parents=True, exist_ok=True)
        processed_density_path.parent.mkdir(parents=True, exist_ok=True)

        padded_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))
        padded_image.paste(image.convert("RGB"), (0, 0))
        padded_image.save(processed_image_path)

    density = np.load(str(density_path)).astype(np.float32, copy=False)
    padded_density = np.zeros((new_height, new_width), dtype=np.float32)
    copy_height = min(height, density.shape[0])
    copy_width = min(width, density.shape[1])
    padded_density[:copy_height, :copy_width] = density[:copy_height, :copy_width]
    np.save(str(processed_density_path), padded_density)

    return processed_image_path, processed_density_path, True


def write_list(list_dir, name, entries, repo_root):
    list_dir.mkdir(parents=True, exist_ok=True)
    target = list_dir / name
    with target.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(entry + "\n")

    print(f"Wrote {display_path(target, repo_root)} ({len(entries)} entries)")
    return target


def main():
    args = parse_args()
    repo_root = Path.cwd().resolve()
    data_root = args.data_root.expanduser().resolve()
    list_dir = args.list_dir or (data_root / "file_lists")
    processed_root = args.processed_root or (data_root / "processed_train")
    list_dir = list_dir.expanduser()
    if not list_dir.is_absolute():
        list_dir = (repo_root / list_dir).resolve()
    processed_root = processed_root.expanduser()
    if not processed_root.is_absolute():
        processed_root = (repo_root / processed_root).resolve()

    if not data_root.exists():
        raise SystemExit(f"Dataset root does not exist: {data_root}")

    part_split_entries = {}
    generated = 0
    existing = 0
    missing_gt = []
    dropped_points = 0
    padded_train_images = 0

    print(f"Dataset root: {display_path(data_root, repo_root)}")
    print(f"List directory: {display_path(list_dir, repo_root)}")
    print(f"Processed train directory: {display_path(processed_root, repo_root)}")
    print(f"Density sigma: {args.sigma}")
    print(f"Overwrite existing density maps: {args.overwrite}")
    print(f"MATLAB one-based coordinates: {args.matlab_indexed}")
    print(f"Training minimum image size: {args.train_min_size}")

    for part in PARTS:
        for split in SPLITS:
            image_dir = data_root / part / split / "images"
            gt_dir = data_root / part / split / "ground-truth"
            list_key = (part, split)
            part_split_entries[list_key] = []

            if not image_dir.exists():
                print(f"Skipping missing image directory: {display_path(image_dir, repo_root)}")
                continue

            images = sorted(image_dir.glob("*.jpg"), key=image_sort_key)
            print(f"Processing {part}/{split}: {len(images)} images")

            for index, image_path in enumerate(images, start=1):
                mat_path = gt_dir / f"GT_{image_path.stem}.mat"
                density_path = image_path.with_name(f"{image_path.stem}_densitymap.npy")

                if not mat_path.exists():
                    missing_gt.append(display_path(mat_path, repo_root))
                    continue

                if density_path.exists() and not args.overwrite:
                    existing += 1
                else:
                    _, dropped = make_density(
                        image_path,
                        mat_path,
                        density_path,
                        args.sigma,
                        args.matlab_indexed,
                    )
                    generated += 1
                    dropped_points += dropped

                list_image_path = image_path
                if split == "train_data":
                    processed_image_path = (
                        processed_root / part / split / "images" / image_path.name
                    )
                    processed_density_path = processed_image_path.with_name(
                        f"{processed_image_path.stem}_densitymap.npy"
                    )
                    list_image_path, _, was_padded = make_train_ready_image(
                        image_path,
                        density_path,
                        processed_image_path,
                        processed_density_path,
                        args.train_min_size,
                        args.overwrite,
                    )
                    if was_padded:
                        padded_train_images += 1

                part_split_entries[list_key].append(display_path(list_image_path, repo_root))

                if index % 50 == 0 or index == len(images):
                    print(f"  {part}/{split}: {index}/{len(images)}")

            split_name = "train" if split == "train_data" else "test"
            write_list(list_dir, f"{part}_{split_name}.txt", part_split_entries[list_key], repo_root)

    train_entries = []
    test_entries = []
    for part in PARTS:
        train_entries.extend(part_split_entries.get((part, "train_data"), []))
        test_entries.extend(part_split_entries.get((part, "test_data"), []))

    write_list(list_dir, "train.txt", train_entries, repo_root)
    write_list(list_dir, "test.txt", test_entries, repo_root)
    write_list(list_dir, "label_train.txt", train_entries, repo_root)
    write_list(list_dir, "unlabel_train.txt", train_entries, repo_root)
    write_list(list_dir, "val.txt", test_entries, repo_root)

    if missing_gt:
        print("")
        print(f"Warning: skipped {len(missing_gt)} images with missing ground-truth .mat files.")
        for item in missing_gt[:10]:
            print(f"  missing: {item}")
        if len(missing_gt) > 10:
            print(f"  ... {len(missing_gt) - 10} more")

    print("")
    print(f"Generated density maps: {generated}")
    print(f"Existing density maps reused: {existing}")
    print(f"Padded train images used in lists: {padded_train_images}")
    print(f"Dropped out-of-bounds points: {dropped_points}")
    print("")
    print("Suggested arguments for main.py:")
    print(f"  --label-file-list {display_path(list_dir / 'label_train.txt', repo_root)}")
    print(f"  --unlabel-file-list {display_path(list_dir / 'unlabel_train.txt', repo_root)}")
    print(f"  --val-file-list {display_path(list_dir / 'val.txt', repo_root)}")


if __name__ == "__main__":
    main()
