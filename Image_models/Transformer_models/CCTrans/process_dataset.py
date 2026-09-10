#!/usr/bin/env python3
"""Generate ShanghaiTech Part A/B density maps from MATLAB annotations."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter


DATASET_DIRECTORIES = {
    "sha": "part_A_final",
    "shb": "part_B_final",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create density_maps_constant15/*.npy files for ShanghaiTech "
            "Part A (sha), Part B (shb), or both."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/ShanghaiTech"),
        help="directory containing part_A_final and part_B_final",
    )
    parser.add_argument(
        "--dataset",
        choices=("sha", "shb", "both"),
        default="both",
        help="dataset to process (default: both)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train_data", "test_data"),
        default=("train_data", "test_data"),
        help="dataset splits to process (default: train_data test_data)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=15.0,
        help="fixed Gaussian standard deviation (default: 15)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace density maps that already exist",
    )
    args = parser.parse_args()

    if args.sigma <= 0:
        parser.error("--sigma must be greater than zero")

    return args


def find_annotation_directory(split_directory):
    """Accept the two directory names commonly used by SHA/SHB archives."""
    for name in ("ground-truth", "ground_truth"):
        annotation_directory = split_directory / name
        if annotation_directory.is_dir():
            return annotation_directory

    raise FileNotFoundError(
        "No annotation directory found in {} (expected ground-truth or "
        "ground_truth)".format(split_directory)
    )


def load_points(annotation_path):
    annotation = loadmat(str(annotation_path))
    if "image_info" not in annotation:
        raise KeyError("{} does not contain 'image_info'".format(annotation_path))

    points = np.asarray(annotation["image_info"][0, 0][0, 0][0])
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    return points.reshape(-1, 2).astype(np.float32, copy=False)


def create_density_map(width, height, points, sigma):
    impulses = np.zeros((height, width), dtype=np.float32)
    valid_count = 0

    for x, y in points:
        # ShanghaiTech annotations store coordinates as (x, y).
        x_index = int(x)
        y_index = int(y)
        if 0 <= x_index < width and 0 <= y_index < height:
            # Addition is important when multiple heads occupy the same pixel.
            impulses[y_index, x_index] += 1.0
            valid_count += 1

    if valid_count == 0:
        return impulses, valid_count

    density = gaussian_filter(impulses, sigma=sigma, mode="reflect")

    # Keep the density-map integral equal to the number of valid annotations.
    density_sum = float(density.sum())
    if density_sum > 0:
        density *= valid_count / density_sum

    return density.astype(np.float32, copy=False), valid_count


def image_paths(image_directory):
    paths = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        paths.extend(image_directory.glob(pattern))
    return sorted(set(paths))


def process_split(split_directory, sigma, overwrite):
    image_directory = split_directory / "images"
    if not image_directory.is_dir():
        raise FileNotFoundError("Image directory not found: {}".format(image_directory))

    annotation_directory = find_annotation_directory(split_directory)
    output_directory = split_directory / "density_maps_constant15"
    output_directory.mkdir(parents=True, exist_ok=True)

    images = image_paths(image_directory)
    if not images:
        raise FileNotFoundError("No images found in {}".format(image_directory))

    created = 0
    skipped = 0
    for index, image_path in enumerate(images, start=1):
        output_path = output_directory / (image_path.stem + ".npy")
        if output_path.exists() and not overwrite:
            skipped += 1
            continue

        annotation_path = annotation_directory / ("GT_" + image_path.stem + ".mat")
        if not annotation_path.is_file():
            raise FileNotFoundError(
                "Annotation for {} not found: {}".format(image_path, annotation_path)
            )

        with Image.open(str(image_path)) as image:
            width, height = image.size

        points = load_points(annotation_path)
        density, valid_count = create_density_map(width, height, points, sigma)
        np.save(str(output_path), density)
        created += 1

        invalid_count = len(points) - valid_count
        status = "[{}/{}] {} -> {} (count: {})".format(
            index, len(images), image_path.name, output_path.name, valid_count
        )
        if invalid_count:
            status += " (ignored out-of-bounds points: {})".format(invalid_count)
        print(status)

    print(
        "Finished {}: created {}, skipped {} existing map(s).".format(
            split_directory, created, skipped
        )
    )


def main():
    args = parse_args()
    datasets = DATASET_DIRECTORIES if args.dataset == "both" else {
        args.dataset: DATASET_DIRECTORIES[args.dataset]
    }

    for dataset_name, directory_name in datasets.items():
        dataset_directory = args.data_root / directory_name
        if not dataset_directory.is_dir():
            raise FileNotFoundError(
                "{} dataset directory not found: {}".format(
                    dataset_name, dataset_directory
                )
            )

        for split_name in args.splits:
            process_split(
                dataset_directory / split_name,
                sigma=args.sigma,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
