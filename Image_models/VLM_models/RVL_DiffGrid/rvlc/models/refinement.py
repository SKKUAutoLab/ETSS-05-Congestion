#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refinement module Phi(.) (paper §III-C, Eq. 12).

Menggabungkan fitur latent multi-scale F = {F64, F32, F16} dari decoder U-Net
menjadi satu peta fitur halus Fref pada resolusi 64x64, menekan noise difusi
single-step.

Channel input tiap skala bisa berbeda (SD1.5: ~320/640/1280), maka dipakai
LazyConv2d agar in_channels otomatis terdeteksi saat forward pertama.
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class Refinement(nn.Module):
    def __init__(self, out_dim: int = 256, target_scale: int = 64):
        super().__init__()
        self.out_dim = out_dim
        self.target_scale = target_scale

        # proyeksi per-skala (lazy -> channel-agnostic)
        self.proj = nn.ModuleDict({
            "64": nn.LazyConv2d(out_dim, kernel_size=1),
            "32": nn.LazyConv2d(out_dim, kernel_size=1),
            "16": nn.LazyConv2d(out_dim, kernel_size=1),
        })

        # refinement setelah penjumlahan multi-scale
        self.refine = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.GroupNorm(8, out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.GroupNorm(8, out_dim),
            nn.GELU(),
        )

    def forward(self, feats: Dict[int, torch.Tensor]) -> torch.Tensor:
        acc = None
        for s in (64, 32, 16):
            if s not in feats:
                continue
            x = self.proj[str(s)](feats[s])
            if x.shape[-1] != self.target_scale:
                x = F.interpolate(
                    x, size=(self.target_scale, self.target_scale),
                    mode="bilinear", align_corners=False,
                )
            acc = x if acc is None else acc + x
        if acc is None:
            raise ValueError("[Refinement] feats kosong; tidak ada skala {64,32,16}.")
        return self.refine(acc)  # [B, out_dim, 64, 64]


if __name__ == "__main__":
    m = Refinement(256)
    feats = {
        64: torch.randn(2, 320, 64, 64),
        32: torch.randn(2, 640, 32, 32),
        16: torch.randn(2, 1280, 16, 16),
    }
    print("Fref:", tuple(m(feats).shape))
