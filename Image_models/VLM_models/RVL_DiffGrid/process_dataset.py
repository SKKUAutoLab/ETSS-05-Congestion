#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normalize ShanghaiTech Part A and Part B into the layout shown in org.txt.

For each selected part, this script writes:

  <part_root>/
  |-- train/images/*.jpg
  |-- test/images/*.jpg
  `-- Processed_JSON/
      |-- annotations_train.json
      `-- annotations_test.json

The JSON files contain image paths, point annotations, and scalar counts.
Density maps are intentionally not precomputed here; training builds them
from points with the configured density mode.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("Missing dependency: numpy. Install the project requirements first.") from exc

try:
    from scipy.io import loadmat
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: scipy. Install it with `pip install scipy` to read ShanghaiTech .mat files."
    ) from exc


TEXT_PROMPT = "crowd of people"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Process ShanghaiTech Part A/B into image folders and annotation JSON files."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=script_dir / "datasets" / "ShanghaiTech",
        help="ShanghaiTech source root containing part_A and part_B.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output root. Defaults to --src. Each processed part is written under part_A/part_B.",
    )
    parser.add_argument(
        "--part",
        default="all",
        help="A, B, part_A, part_B, comma-separated values, or all.",
    )
    image_mode = parser.add_mutually_exclusive_group()
    image_mode.add_argument(
        "--copy",
        action="store_const",
        const="copy",
        dest="image_mode",
        help="Copy images instead of creating symlinks.",
    )
    image_mode.add_argument(
        "--hardlink",
        action="store_const",
        const="hardlink",
        dest="image_mode",
        help="Hardlink images instead of creating symlinks.",
    )
    image_mode.add_argument(
        "--no-images",
        action="store_const",
        const="none",
        dest="image_mode",
        help="Only write Processed_JSON files.",
    )
    parser.set_defaults(image_mode="symlink")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing generated image links/files.",
    )
    return parser.parse_args()


def natural_key(path: Path) -> list[Any]:
    parts = re.split(r"([0-9]+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def normalize_parts(value: str) -> list[str]:
    if value.lower() == "all":
        return ["A", "B"]

    normalized = []
    for raw in value.split(","):
        item = raw.strip().replace("-", "_")
        if not item:
            continue

        upper = item.upper()
        if upper in {"A", "PART_A"}:
            normalized.append("A")
        elif upper in {"B", "PART_B"}:
            normalized.append("B")
        else:
            raise SystemExit(f"Invalid --part value: {raw}")

    if not normalized:
        raise SystemExit("No parts selected.")

    return list(dict.fromkeys(normalized))


def read_mat(path: Path) -> dict[str, Any]:
    try:
        return loadmat(path, simplify_cells=True)
    except TypeError:
        return loadmat(path, squeeze_me=True, struct_as_record=False)


def unwrap_single(value: Any) -> Any:
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.item()
    return value


def as_points(value: Any) -> list[list[float]] | None:
    value = unwrap_single(value)
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None

    if arr.size == 0:
        return []

    if arr.ndim == 1:
        if arr.shape[0] != 2:
            return None
        arr = arr.reshape(1, 2)
    elif arr.ndim > 2:
        if arr.shape[-1] != 2:
            return None
        arr = arr.reshape(-1, 2)

    if arr.ndim != 2:
        return None
    if arr.shape[1] == 2:
        pass
    elif arr.shape[0] == 2:
        arr = arr.T
    elif arr.shape[1] > 2:
        arr = arr[:, :2]
    else:
        return None

    arr = arr[np.isfinite(arr).all(axis=1)]
    return [[float(x), float(y)] for x, y in arr]


def find_points(value: Any, depth: int = 0) -> list[list[float]] | None:
    if depth > 12:
        return None

    value = unwrap_single(value)

    if isinstance(value, dict):
        for key in ("location", "annPoints", "ann_points", "points"):
            if key in value:
                points = as_points(value[key])
                if points is not None:
                    return points
        for nested in value.values():
            points = find_points(nested, depth + 1)
            if points is not None:
                return points
        return None

    if hasattr(value, "_fieldnames"):
        for key in ("location", "annPoints", "ann_points", "points"):
            if hasattr(value, key):
                points = as_points(getattr(value, key))
                if points is not None:
                    return points
        for key in value._fieldnames:
            points = find_points(getattr(value, key), depth + 1)
            if points is not None:
                return points
        return None

    if isinstance(value, (list, tuple)):
        for nested in value:
            points = find_points(nested, depth + 1)
            if points is not None:
                return points
        return None

    if isinstance(value, np.ndarray):
        points = as_points(value)
        if points is not None:
            return points
        if value.dtype == object:
            for nested in value.flat:
                points = find_points(nested, depth + 1)
                if points is not None:
                    return points

    return None


def load_points(mat_path: Path) -> list[list[float]]:
    data = read_mat(mat_path)
    root = data.get("image_info", data)
    points = find_points(root)
    if points is None:
        raise ValueError(f"Could not find point annotations in {mat_path}")
    return points


def materialize_image(src: Path, dst: Path, image_mode: str, force: bool) -> None:
    if image_mode == "none":
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not force:
            return
        if dst.is_dir():
            raise IsADirectoryError(f"Refusing to replace directory: {dst}")
        dst.unlink()

    if image_mode == "symlink":
        rel_src = os.path.relpath(src, dst.parent)
        os.symlink(rel_src, dst)
    elif image_mode == "copy":
        shutil.copy2(src, dst)
    elif image_mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"Unsupported image mode: {image_mode}")


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
        handle.write("\n")
    tmp_path.replace(path)


def process_split(
    src_part: Path,
    out_part: Path,
    split: str,
    image_mode: str,
    force: bool,
) -> tuple[int, int, list[str]]:
    src_split = src_part / f"{split}_data"
    src_images = src_split / "images"
    src_gt = src_split / "ground-truth"
    out_images = out_part / split / "images"

    if not src_images.is_dir():
        raise SystemExit(f"Missing image directory: {src_images}")
    if not src_gt.is_dir():
        raise SystemExit(f"Missing ground-truth directory: {src_gt}")

    rows = []
    warnings = []
    for image_path in sorted(src_images.glob("*.jpg"), key=natural_key):
        gt_path = src_gt / f"GT_{image_path.stem}.mat"
        if not gt_path.exists():
            warnings.append(f"missing ground truth for {image_path.name}")
            points = []
        else:
            points = load_points(gt_path)

        dst_image = out_images / image_path.name
        materialize_image(image_path, dst_image, image_mode, force)
        rows.append(
            {
                "image_path": f"{split}/images/{image_path.name}",
                "gt_count": len(points),
                "points": points,
                "text": TEXT_PROMPT,
            }
        )

    json_path = out_part / "Processed_JSON" / f"annotations_{split}.json"
    write_json(json_path, rows)
    return len(rows), sum(len(row["points"]) for row in rows), warnings


def process_part(
    letter: str,
    src_root: Path,
    out_root: Path,
    image_mode: str,
    force: bool,
) -> None:
    src_part = src_root / f"part_{letter}"
    out_part = out_root / f"part_{letter}"
    if not src_part.is_dir():
        raise SystemExit(f"Missing source part directory: {src_part}")

    summary = {}
    warnings = []
    for split in ("train", "test"):
        count, total_points, split_warnings = process_split(
            src_part=src_part,
            out_part=out_part,
            split=split,
            image_mode=image_mode,
            force=force,
        )
        summary[split] = (count, total_points)
        warnings.extend(f"{split}: {msg}" for msg in split_warnings)

    print(f"Processed part_{letter} -> {out_part}")
    for split, (count, total_points) in summary.items():
        print(f"  {split}: {count} images, {total_points} annotated points")
    for warning in warnings:
        print(f"  warning: {warning}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    src_root = args.src.expanduser().resolve()
    out_root = (args.out or args.src).expanduser().resolve()

    if not src_root.is_dir():
        raise SystemExit(f"Missing source root: {src_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    for part in normalize_parts(args.part):
        process_part(
            letter=part,
            src_root=src_root,
            out_root=out_root,
            image_mode=args.image_mode,
            force=args.force,
        )


if __name__ == "__main__":
    main()
