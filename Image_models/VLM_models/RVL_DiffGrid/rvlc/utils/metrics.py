#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metrik counting: MAE & RMSE."""
import numpy as np


def mae_rmse(errors):
    """errors: array |pred-gt|. return (mae, rmse)."""
    errors = np.asarray(errors, dtype=np.float64)
    if errors.size == 0:
        return float("inf"), float("inf")
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return mae, rmse
