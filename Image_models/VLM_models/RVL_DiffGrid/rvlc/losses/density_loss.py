#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Density regression loss L_dens (paper Eq. 20).

Standard l2 (MSE) antara density prediksi dan GT density map.
GT dibuat pada resolusi output regresor (lihat data/density_map.py) agar tidak
perlu resize. Jika ukuran beda, GT di-resize sum-preserving.
"""
import torch
import torch.nn.functional as F


def density_loss(pred_density: torch.Tensor, gt_density: torch.Tensor) -> torch.Tensor:
    """
    pred_density : [B,1,H,W]
    gt_density   : [B,1,Hg,Wg]
    """
    if gt_density.shape[-2:] != pred_density.shape[-2:]:
        # resize sum-preserving: skala dgn rasio luas agar total count tetap
        before = gt_density.flatten(1).sum(dim=1, keepdim=True)
        gt_density = F.interpolate(
            gt_density, size=pred_density.shape[-2:], mode="bilinear", align_corners=False
        )
        after = gt_density.flatten(1).sum(dim=1, keepdim=True).clamp(min=1e-6)
        gt_density = gt_density * (before / after).view(-1, 1, 1, 1)

    return F.mse_loss(pred_density, gt_density)
