#!/usr/bin/env python
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

try:
    import scipy.io as sio
except ImportError as exc:
    raise SystemExit(
        "scipy is required to convert ShanghaiTech .mat annotations to JSON labels"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "datasets" / "ShanghaiTech"
PARTS = ("part_A", "part_B")
SPLITS = ("train_data", "test_data")


def require_dir(path: Path) -> None:
    if not path.is_dir():
        raise SystemExit(f"Required directory not found: {path}")


def ensure_ground_truth_dir(split_dir: Path) -> None:
    raw_dir = split_dir / "ground-truth"
    compat_dir = split_dir / "ground_truth"

    if compat_dir.exists() or compat_dir.is_symlink():
        if not compat_dir.is_dir():
            raise SystemExit(f"Existing ground_truth path is not a directory: {compat_dir}")
        return

    require_dir(raw_dir)

    try:
        os.symlink("ground-truth", compat_dir)
        print(f"Linked {compat_dir} -> ground-truth")
    except OSError:
        shutil.copytree(raw_dir, compat_dir)
        print(f"Copied {raw_dir} to {compat_dir}")


def shanghaitech_points(mat_path: Path) -> list[dict[str, float]]:
    mat = sio.loadmat(str(mat_path))
    if "image_info" not in mat:
        raise ValueError(f"missing image_info in {mat_path}")

    points = mat["image_info"][0][0][0][0][0]
    return [{"x": float(point[0]), "y": float(point[1])} for point in points]


def prepare_layout() -> None:
    require_dir(DATA_ROOT)

    for part in PARTS:
        part_dir = DATA_ROOT / part
        require_dir(part_dir)

        for split in SPLITS:
            split_dir = part_dir / split
            require_dir(split_dir)
            require_dir(split_dir / "images")
            ensure_ground_truth_dir(split_dir)
            (split_dir / "labels").mkdir(exist_ok=True)


def convert_labels() -> tuple[int, int]:
    total = 0
    written = 0

    for part in PARTS:
        for split in SPLITS:
            split_dir = DATA_ROOT / part / split
            gt_dir = split_dir / "ground_truth"
            labels_dir = split_dir / "labels"

            mat_files = sorted(gt_dir.glob("*.mat"))
            if not mat_files:
                raise SystemExit(f"No .mat files found in {gt_dir}")

            for mat_path in mat_files:
                if not mat_path.name.startswith("GT_"):
                    raise SystemExit(f"Unexpected annotation filename: {mat_path}")

                image_stem = mat_path.stem[3:]
                image_path = split_dir / "images" / f"{image_stem}.jpg"
                if not image_path.is_file():
                    raise SystemExit(
                        f"Missing image for annotation {mat_path}: expected {image_path}"
                    )

                label_path = labels_dir / f"{image_stem}.json"
                payload = json.dumps(
                    shanghaitech_points(mat_path),
                    separators=(",", ":"),
                )

                if not label_path.exists() or label_path.read_text() != payload:
                    label_path.write_text(payload)
                    written += 1
                total += 1

            print(f"{part}/{split}: {len(mat_files)} annotations processed")

    return total, written


def main() -> None:
    if len(sys.argv) > 1:
        raise SystemExit(
            f"This script always processes {DATA_ROOT} and does not accept an output path."
        )

    print(f"Preparing ShanghaiTech dataset at {DATA_ROOT}")
    prepare_layout()
    total, written = convert_labels()
    print(f"JSON labels ready: {total} files checked, {written} files written")
    print("ShanghaiTech dataset processing complete")


if __name__ == "__main__":
    main()
