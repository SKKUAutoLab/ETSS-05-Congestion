#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global counting head (paper §III-B).

Memproses Ffused -> coarse global count C_global (skalar).
Hanya dipakai saat training (auxiliary), dibuang saat inference.

CATATAN: ADR (L_ADR, Eq.6) TIDAK dihitung di sini. Di paper L_ADR mengikat
C_global dengan integral density map (cross-branch), jadi dihitung di level
model/loss, bukan di dalam head. Head ini murni regresor skalar.
"""
import torch
import torch.nn as nn


class GlobalCountHead(nn.Module):
    def __init__(self, in_dim: int = 768, hidden_ratio: float = 0.5, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        hidden = max(64, int(in_dim * hidden_ratio))
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        if fused.dim() == 3:
            fused = fused.mean(1)
        if fused.shape[-1] != self.in_dim:
            raise ValueError(
                f"[GlobalCountHead] dim mismatch: got {fused.shape[-1]}, expected {self.in_dim}"
            )
        # softplus -> count non-negatif
        return torch.nn.functional.softplus(self.mlp(fused).squeeze(-1))  # [B]


if __name__ == "__main__":
    h = GlobalCountHead(768)
    print("count:", tuple(h(torch.randn(8, 768)).shape))
