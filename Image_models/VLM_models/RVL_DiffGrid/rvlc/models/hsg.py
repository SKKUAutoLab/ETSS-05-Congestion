#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hierarchical Semantic Grids (HSG) — paper §III-D, Eq. 13-14.

Non-parametric (sesuai paper: bobot uniform tetap, learnable weights justru
+0.30 MAA -> dibuang).

Langkah:
  1) Tiap attention map skala s in {64,32,16} di-resize bilinear ke 16x16 (Eq.13).
  2) Min-max normalisasi PER-IMAGE ke [0,1].
  3) Weighted sum dengan bobot uniform ws = 1/3 (Eq.14) -> S [B,1,16,16].
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class HSG(nn.Module):
    def __init__(self, grid_size: int = 16, scales=(64, 32, 16)):
        super().__init__()
        self.grid_size = grid_size
        self.scales = scales

    @staticmethod
    def _minmax_per_image(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # x: [B,1,g,g] -> normalisasi per-image
        B = x.shape[0]
        flat = x.view(B, -1)
        mn = flat.min(dim=1, keepdim=True).values
        mx = flat.max(dim=1, keepdim=True).values
        flat = (flat - mn) / (mx - mn + eps)
        return flat.view_as(x)

    def forward(self, attn_maps: Dict[int, torch.Tensor]) -> torch.Tensor:
        grids = []
        present = [s for s in self.scales if s in attn_maps]
        if not present:
            raise ValueError("[HSG] attn_maps kosong untuk skala yang diminta.")
        w = 1.0 / len(present)  # bobot uniform
        for s in present:
            g = F.interpolate(
                attn_maps[s], size=(self.grid_size, self.grid_size),
                mode="bilinear", align_corners=False,
            )
            g = self._minmax_per_image(g)
            grids.append(w * g)
        S = torch.stack(grids, dim=0).sum(0)  # [B,1,16,16]
        return S


if __name__ == "__main__":
    m = HSG()
    attn = {64: torch.rand(2, 1, 64, 64), 32: torch.rand(2, 1, 32, 32), 16: torch.rand(2, 1, 16, 16)}
    print("S:", tuple(m(attn).shape))
