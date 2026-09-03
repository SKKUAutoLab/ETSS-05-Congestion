#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Density regression Gamma(Fref, S) -> D_hat (paper §III-D, Eq. 15).

Concat refined latent feature Fref dengan grid semantik S (di-upsample ke
resolusi Fref), lalu regresor konvolusional ringan menghasilkan density map.
Count akhir = integrasi spasial density (Eq. 16): C = sum_{x,y} D(x,y).

Output density non-negatif (softplus).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DensityRegressor(nn.Module):
    def __init__(self, in_dim: int = 256, hidden: int = 128):
        super().__init__()
        # +1 channel untuk grid S
        self.head = nn.Sequential(
            nn.Conv2d(in_dim + 1, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden // 2, 1, 1),
        )

    def forward(self, fref: torch.Tensor, grid_S: torch.Tensor) -> torch.Tensor:
        # samakan resolusi S dengan Fref
        if grid_S.shape[-2:] != fref.shape[-2:]:
            grid_S = F.interpolate(
                grid_S, size=fref.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([fref, grid_S], dim=1)
        density = F.softplus(self.head(x))  # [B,1,H,W] >= 0
        return density

    @staticmethod
    def integrate(density: torch.Tensor) -> torch.Tensor:
        """C_hat = sum_{x,y} D(x,y) (Eq.16) -> [B]"""
        return density.flatten(1).sum(dim=1)


if __name__ == "__main__":
    m = DensityRegressor(256)
    d = m(torch.randn(2, 256, 64, 64), torch.rand(2, 1, 16, 16))
    print("density:", tuple(d.shape), "count:", m.integrate(d).shape)
