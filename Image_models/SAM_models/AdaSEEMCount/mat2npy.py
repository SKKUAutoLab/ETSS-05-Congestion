#!/usr/bin/env python3
"""Convert ShanghaiTech .mat point annotations to .npy files.

By default this converts Part A test annotations:

    data/ShanghaiTech/part_A/test_data/ground-truth/GT_IMG_1.mat
        -> data/ShanghaiTech/part_A/test_data/images/IMG_1.npy

The output .npy contains an array of shape (num_people, 2), matching the
format expected by run_train_counter.py's evaluate() function.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class PointCandidate:
    path: str
    points: np.ndarray


def natural_key(path: Path) -> tuple[int, str]:
    stem = path.stem.replace("GT_IMG_", "")
    return (int(stem), path.name) if stem.isdigit() else (10**9, path.name)


def load_mat(path: Path) -> dict:
    try:
        return loadmat(path, simplify_cells=True)
    except TypeError:
        return loadmat(path, squeeze_me=True, struct_as_record=False)


def normalize_points(array: np.ndarray) -> np.ndarray | None:
    points = np.asarray(array)

    if points.ndim == 1 and points.size == 2:
        points = points.reshape(1, 2)

    if points.ndim != 2:
        return None

    if points.shape[1] == 2:
        return points.astype(np.float32, copy=False)

    if points.shape[0] == 2 and points.shape[1] != 2:
        return points.T.astype(np.float32, copy=False)

    return None


def collect_point_candidates(value: object, path: str = "image_info") -> list[PointCandidate]:
    candidates: list[PointCandidate] = []

    if isinstance(value, dict):
        for key, child in value.items():
            if key.startswith("__"):
                continue
            candidates.extend(collect_point_candidates(child, f"{path}.{key}"))
        return candidates

    if hasattr(value, "_fieldnames"):
        for key in value._fieldnames:
            candidates.extend(collect_point_candidates(getattr(value, key), f"{path}.{key}"))
        return candidates

    if isinstance(value, np.void) and value.dtype.names:
        for key in value.dtype.names:
            candidates.extend(collect_point_candidates(value[key], f"{path}.{key}"))
        return candidates

    if isinstance(value, (list, tuple)):
        for idx, child in enumerate(value):
            candidates.extend(collect_point_candidates(child, f"{path}[{idx}]"))
        return candidates

    if isinstance(value, np.ndarray):
        if value.dtype.names:
            for key in value.dtype.names:
                candidates.extend(collect_point_candidates(value[key], f"{path}.{key}"))
            return candidates

        if value.dtype == object:
            for idx, child in enumerate(value.flat):
                candidates.extend(collect_point_candidates(child, f"{path}[{idx}]"))
            return candidates

        if np.issubdtype(value.dtype, np.number):
            points = normalize_points(value)
            if points is not None:
                candidates.append(PointCandidate(path=path, points=points))

    return candidates


def extract_points(mat_path: Path) -> np.ndarray:
    mat = load_mat(mat_path)
    if "image_info" not in mat:
        raise KeyError(f"{mat_path} does not contain image_info")

    candidates = collect_point_candidates(mat["image_info"])
    if not candidates:
        raise ValueError(f"Could not find an Nx2 point array in {mat_path}")

    candidates.sort(
        key=lambda item: ("location" in item.path.lower(), item.points.shape[0]),
        reverse=True,
    )
    return np.ascontiguousarray(candidates[0].points, dtype=np.float32)


def convert_annotations(
    root: Path,
    part: str,
    split: str,
    overwrite: bool,
    allow_missing_images: bool,
) -> tuple[int, int]:
    split_dir = root / f"part_{part}" / split
    gt_dir = split_dir / "ground-truth"
    image_dir = split_dir / "images"

    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth directory not found: {gt_dir}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    converted = 0
    skipped = 0

    for mat_path in sorted(gt_dir.glob("GT_IMG_*.mat"), key=natural_key):
        image_name = mat_path.name.replace("GT_", "").replace(".mat", ".jpg")
        image_path = image_dir / image_name
        npy_path = image_path.with_suffix(".npy")

        if not image_path.exists() and not allow_missing_images:
            print(f"skip missing image: {image_path}")
            skipped += 1
            continue

        if npy_path.exists() and not overwrite:
            skipped += 1
            continue

        points = extract_points(mat_path)
        np.save(npy_path, points)
        print(f"saved {npy_path} ({len(points)} points)")
        converted += 1

    return converted, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ShanghaiTech GT_IMG_*.mat files to IMG_*.npy point arrays."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/ShanghaiTech"),
        help="ShanghaiTech dataset root.",
    )
    parser.add_argument(
        "--part",
        choices=("A", "B"),
        default="A",
        help="ShanghaiTech part to convert.",
    )
    parser.add_argument(
        "--split",
        choices=("train_data", "test_data"),
        default="test_data",
        help="Dataset split to convert.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npy files.",
    )
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Write .npy files even when the matching .jpg is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    converted, skipped = convert_annotations(
        root=args.root,
        part=args.part,
        split=args.split,
        overwrite=args.overwrite,
        allow_missing_images=args.allow_missing_images,
    )
    print(f"done: converted={converted}, skipped={skipped}")


if __name__ == "__main__":
    main()
