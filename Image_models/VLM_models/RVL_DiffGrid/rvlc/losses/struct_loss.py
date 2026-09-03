#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structural regularization (paper §III-E, Eq. 17-19).

L_struct = lLRRC * L_LRRC + lgrid * L_grid
  - L_LRRC : TV anisotropik orde-1 pada grid 16x16 S (Eq.17).
  - L_grid : |G_k(D_hat) - G_k(D_gt)| pada grid g x g (Eq.18),
             G_k = integral density dalam sel ke-k.
"""
import torch
import torch.nn.functional as F


def lrrc_loss(S: torch.Tensor) -> torch.Tensor:
    """
    S: [B,1,g,g] grid semantik. TV anisotropik orde-1 (Eq.17).
    """
    dh = torch.abs(S[:, :, 1:, :] - S[:, :, :-1, :])   # beda vertikal
    dw = torch.abs(S[:, :, :, 1:] - S[:, :, :, :-1])   # beda horizontal
    return dh.mean() + dw.mean()


def _grid_integrate(D: torch.Tensor, g: int = 16) -> torch.Tensor:
    """Integral (sum) density dalam tiap sel grid g x g -> [B,1,g,g]."""
    # avg_pool * luas sel = sum dalam sel
    H, W = D.shape[-2:]
    pooled = F.adaptive_avg_pool2d(D, output_size=(g, g))
    cell_area = (H / g) * (W / g)
    return pooled * cell_area


def grid_loss(pred_density: torch.Tensor, gt_density: torch.Tensor, g: int = 16) -> torch.Tensor:
    """
    L_grid (Eq.18): konsistensi distribusi density per-sel terhadap GT.
    """
    gk_pred = _grid_integrate(pred_density, g)
    gk_gt = _grid_integrate(gt_density, g)
    return F.l1_loss(gk_pred, gk_gt)


def struct_loss(
    S: torch.Tensor,
    pred_density: torch.Tensor,
    gt_density: torch.Tensor = None,
    lambda_lrrc: float = 0.05,
    lambda_grid: float = 0.05,
    grid_size: int = 16,
):
    l_lrrc = lrrc_loss(S)
    if gt_density is not None and lambda_grid > 0:
        l_grid = grid_loss(pred_density, gt_density, g=grid_size)
    else:
        l_grid = torch.tensor(0.0, device=S.device)

    total = lambda_lrrc * l_lrrc + lambda_grid * l_grid
    return total, {"l_lrrc": l_lrrc.detach(), "l_grid": l_grid.detach()}
