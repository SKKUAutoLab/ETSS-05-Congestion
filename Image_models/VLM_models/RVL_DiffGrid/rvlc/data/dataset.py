#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CountDataset — loader fleksibel untuk ShanghaiTech / UCF / JHU.

JSON per-item boleh memakai:
  - gambar : "image_path" | "img_path" | "path" | "image" | "img"
             atau "image_id" | "id" | "name" (-> IMG_xxx.jpg)
  - count  : "gt_count" | "gt" | "count"
  - titik  : "points" | "ann_points"  (list [x,y]) -> dipakai utk GT density
             dan juga sebagai sumber count bila count scalar tidak ada.

__getitem__ mengembalikan PIL + count + points + ukuran asli (utk density map).
"""
import json
import logging
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


IMG_KEYS = ["image_path", "img_path", "path", "image", "img"]
GT_SCALAR = ["gt_count", "gt", "count"]
GT_POINTS = ["points", "ann_points"]
ID_KEYS = ["image_id", "id", "name"]


class CountDataset(Dataset):
    def __init__(self, json_path: str, img_root: Optional[str] = None):
        assert os.path.exists(json_path), f"[ERROR] JSON not found: {json_path}"
        with open(json_path, "r") as f:
            data = json.load(f)

        if img_root is None:
            img_root = os.path.abspath(os.path.join(os.path.dirname(json_path), ".."))
        self.img_root = img_root

        base = os.path.basename(json_path).lower()
        self.split = "train" if "train" in base else ("test" if "test" in base else None)

        logging.info(f"[DATASET] json={json_path} img_root={self.img_root} split={self.split}")

        self.items = []
        missing = 0
        for it in data:
            # ---- points & count ----
            points = None
            for k in GT_POINTS:
                if k in it and isinstance(it[k], list):
                    points = np.asarray(it[k], dtype=np.float32).reshape(-1, 2)
                    break
            gt = None
            for k in GT_SCALAR:
                if k in it and not isinstance(it[k], (list, dict)):
                    gt = float(it[k]); break
            if gt is None:
                gt = float(len(points)) if points is not None else 0.0

            # ---- resolve image ----
            rel = None
            for k in IMG_KEYS:
                if k in it and it[k] is not None:
                    rel = str(it[k]); break
            if rel is None:
                for k in ID_KEYS:
                    if k in it:
                        rel = str(it[k])
                        if not rel.lower().endswith(".jpg"):
                            rel += ".jpg"
                        break
            if rel is None:
                missing += 1; continue

            fname = os.path.basename(rel)
            cands = []
            if os.path.isabs(rel):
                cands.append(rel)
            cands.append(os.path.join(self.img_root, rel))
            if self.split:
                cands.append(os.path.join(self.img_root, self.split, rel))
                cands.append(os.path.join(self.img_root, self.split, "images", fname))
            else:
                for sp in ("train", "test"):
                    cands.append(os.path.join(self.img_root, sp, rel))
                    cands.append(os.path.join(self.img_root, sp, "images", fname))

            img_path = next((c for c in cands if os.path.exists(c)), None)
            if img_path is None:
                missing += 1; continue

            self.items.append({
                "image_path": img_path,
                "text": it.get("text", "crowd of people"),
                "gt_count": gt,
                "points": points,
            })

        if missing:
            logging.warning(f"[WARN] Skipped {missing} samples (image not found).")
        logging.info(f"[INFO] Loaded {len(self.items)} samples from {json_path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = Image.open(it["image_path"]).convert("RGB")
        W, H = img.size
        return {
            "image_pil": img,
            "text": it["text"],
            "gt_count": torch.tensor(it["gt_count"], dtype=torch.float32),
            "points": it["points"],          # np.ndarray [N,2] atau None
            "orig_size": (W, H),
        }


def collate_fn(batch):
    return {
        "images_pil": [b["image_pil"] for b in batch],
        "texts": [b["text"] for b in batch],
        "gt": torch.stack([b["gt_count"] for b in batch], dim=0),
        "points": [b["points"] for b in batch],
        "orig_size": [b["orig_size"] for b in batch],
    }
