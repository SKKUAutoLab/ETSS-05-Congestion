#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ground-truth density map generation (paper §IV-A).

- ShanghaiTech / UCF : adaptive Gaussian kernel (sigma dari kNN antar-titik).
- JHU-Crowd++        : fixed Gaussian kernel.

Density dibangun langsung pada resolusi output (default 64x64, = resolusi
regresor), dengan total massa == jumlah titik (count).
"""
from typing import Optional, Tuple

import numpy as np

try:
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except Exception:  # noqa
    _HAS_SCIPY = False


def build_density_map(
    points: Optional[np.ndarray],     # [N,2] (x,y) dalam piksel gambar asli
    orig_size: Tuple[int, int],       # (W, H) gambar asli
    out_size: Tuple[int, int] = (64, 64),  # (Hg, Wg)
    mode: str = "adaptive",           # "adaptive" | "fixed"
    fixed_sigma: float = 4.0,
    knn: int = 3,
    beta: float = 0.3,
) -> np.ndarray:
    """
    return: density [Hg, Wg] float32, sum == N.
    """
    Hg, Wg = out_size
    density = np.zeros((Hg, Wg), dtype=np.float32)

    if points is None or len(points) == 0:
        return density

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    W, H = orig_size
    sx, sy = Wg / max(W, 1), Hg / max(H, 1)

    # skala titik ke grid output
    gx = np.clip((pts[:, 0] * sx).astype(int), 0, Wg - 1)
    gy = np.clip((pts[:, 1] * sy).astype(int), 0, Hg - 1)

    if not _HAS_SCIPY:
        # fallback: tanpa scipy, taruh impuls (sum tetap = N)
        for x, y in zip(gx, gy):
            density[y, x] += 1.0
        return density

    if mode == "fixed":
        pt_map = np.zeros((Hg, Wg), dtype=np.float32)
        for x, y in zip(gx, gy):
            pt_map[y, x] += 1.0
        density = gaussian_filter(pt_map, sigma=fixed_sigma, mode="constant")
    else:  # adaptive
        coords = np.stack([gx, gy], axis=1).astype(np.float32)
        n = len(coords)
        if n > 1:
            tree = cKDTree(coords)
            k = min(knn + 1, n)
            dists, _ = tree.query(coords, k=k)
            # rata-rata jarak tetangga (abaikan diri sendiri di kolom 0)
            mean_d = dists[:, 1:].mean(axis=1) if k > 1 else np.ones(n)
        else:
            mean_d = np.array([min(Hg, Wg) / 4.0], dtype=np.float32)

        for (x, y), md in zip(coords.astype(int), mean_d):
            sigma = max(beta * float(md), 0.5)
            impulse = np.zeros((Hg, Wg), dtype=np.float32)
            impulse[y, x] = 1.0
            density += gaussian_filter(impulse, sigma=sigma, mode="constant")

    # jaga total massa == N
    s = density.sum()
    if s > 1e-8:
        density *= (len(pts) / s)
    return density.astype(np.float32)
