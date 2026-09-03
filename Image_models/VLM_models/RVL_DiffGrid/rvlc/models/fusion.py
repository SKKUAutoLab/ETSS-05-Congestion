#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fusion module phi(Fv, Et) -> Ffused (paper §III-A, Eq. 3).

Paper: "phi(.) denotes a linear projection or lightweight cross-attention module".
Disediakan dua mode ringan:
  - "linear": LayerNorm(img|txt) -> concat -> Linear -> GELU   (default, hemat param)
  - "gated" : gating txt terhadap img (residual)
Output: vektor fused [B, D] yang dipakai global head DAN context diffusion.
"""
import torch
import torch.nn as nn


class Fusion(nn.Module):
    def __init__(self, dim: int = 768, mode: str = "linear"):
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == "linear":
            self.norm_img = nn.LayerNorm(dim)
            self.norm_txt = nn.LayerNorm(dim)
            self.fc = nn.Linear(dim * 2, dim)
            self.act = nn.GELU()
        elif mode == "gated":
            self.proj = nn.Linear(dim, dim)
            self.gate = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
                nn.Linear(dim, dim), nn.Sigmoid(),
            )
        else:
            raise ValueError(f"mode fusion tidak dikenal: {mode}")

    def forward(self, img_feat: torch.Tensor, txt_feat: torch.Tensor) -> torch.Tensor:
        # terima [B,D] atau [B,N,D]
        if img_feat.dim() == 3:
            img_feat = img_feat.mean(1)
        if txt_feat.dim() == 3:
            txt_feat = txt_feat.mean(1)

        if self.mode == "linear":
            x = torch.cat([self.norm_img(img_feat), self.norm_txt(txt_feat)], dim=-1)
            return self.act(self.fc(x))
        else:  # gated
            x = self.proj(img_feat)
            g = self.gate(txt_feat)
            return x * g + img_feat * (1 - g)


if __name__ == "__main__":
    f = Fusion(768, "linear")
    y = f(torch.randn(4, 768), torch.randn(4, 768))
    print("fused:", tuple(y.shape))
