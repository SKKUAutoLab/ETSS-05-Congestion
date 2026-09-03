#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global counting loss (paper §III-B, Eq. 4-6).

L_global = L_Huber + lrel * L_rel + lADR * L_ADR
  - L_Huber : smooth_l1 antara C_global dan C_gt (Eq.4)
  - L_rel   : |C_global - C_gt| / (C_gt + eps)              (Eq.5)
  - L_ADR   : |C_global - sum_xy D_hat(x,y)|                (Eq.6, cross-branch)
"""
import torch
import torch.nn.functional as F


def global_loss(
    global_count: torch.Tensor,   # [B]  C_global
    density_count: torch.Tensor,  # [B]  sum D_hat
    gt_count: torch.Tensor,       # [B]
    lambda_rel: float = 0.03,
    lambda_adr: float = 0.01,
    eps: float = 1e-6,
):
    gt = gt_count.float()

    l_huber = F.smooth_l1_loss(global_count, gt)
    l_rel = torch.mean(torch.abs(global_count - gt) / (gt + eps))
    l_adr = torch.mean(torch.abs(global_count - density_count))

    total = l_huber + lambda_rel * l_rel + lambda_adr * l_adr
    return total, {
        "l_huber": l_huber.detach(),
        "l_rel": l_rel.detach(),
        "l_adr": l_adr.detach(),
    }
