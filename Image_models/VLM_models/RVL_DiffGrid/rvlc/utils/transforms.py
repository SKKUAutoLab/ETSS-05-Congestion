#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Target transform (opsional) untuk stabilitas regresi count.

CATATAN: jalur inference RVL-DiffGrid memakai integrasi density (count mentah),
jadi transform ini OPSIONAL dan default 'none'. Disediakan untuk eksperimen.
"""
import torch


def get_target_transform(name: str = "none"):
    name = (name or "none").lower()
    if name == "log":
        f = lambda x: torch.log1p(torch.clamp(x, min=0.0))
        finv = lambda y: torch.expm1(y)
    elif name == "sqrt":
        f = lambda x: torch.sqrt(torch.clamp(x, min=0.0))
        finv = lambda y: torch.clamp(y, min=0.0) ** 2
    else:
        f = lambda x: x
        finv = lambda y: y
    return f, finv
