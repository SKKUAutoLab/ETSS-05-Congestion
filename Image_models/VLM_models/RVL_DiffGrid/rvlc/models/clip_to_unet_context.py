#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP -> U-Net cross-attention context (paper §III-C, Eq.9).

Paper: "The fused CLIP embedding is projected through a learnable linear layer
to the diffusion context dimension (768) and reshaped into a token sequence of
length N = 77". Text encoder SD di-bypass; tensor inilah yang dipakai sebagai
encoder_hidden_states untuk frozen U-Net.

Catatan param-budget: paper mengklaim total trainable ~0.30M. Linear penuh
(clip_dim x 768) saja sudah ~0.59M utk clip_dim=768. Bila ingin menekan param,
pakai mode "lowrank" (faktorisasi) atau kecilkan token efektif.
"""

import torch
import torch.nn as nn


class CLIPToUNetContext(nn.Module):
    def __init__(
        self,
        clip_dim: int = 768,
        context_dim: int = 768,   # cross_attention_dim SD v1.5
        num_tokens: int = 77,     # default CLIP context length SD
        mode: str = "expand",     # "expand" | "lowrank"
        rank: int = 64,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.context_dim = context_dim
        self.mode = mode

        if mode == "expand":
            # proyeksi 1 vektor -> 1 vektor context, lalu di-broadcast ke 77 token
            self.proj = nn.Linear(clip_dim, context_dim)
        elif mode == "lowrank":
            # faktorisasi utk menekan jumlah parameter
            self.proj = nn.Sequential(
                nn.Linear(clip_dim, rank, bias=False),
                nn.Linear(rank, context_dim),
            )
        else:
            raise ValueError(f"mode tidak dikenal: {mode}")

        # token positional embedding ringan (opsional) supaya 77 token tidak identik
        self.token_embed = nn.Parameter(torch.zeros(1, num_tokens, context_dim))
        nn.init.normal_(self.token_embed, std=0.02)

    def forward(self, fused_clip: torch.Tensor) -> torch.Tensor:
        """
        fused_clip: [B, clip_dim]  (output FusionModule / phi(Fv,Et))
        return    : [B, num_tokens, context_dim]
        """
        if fused_clip.dim() == 3:
            fused_clip = fused_clip.mean(1)  # jaga-jaga kalau [B,N,D]

        ctx = self.proj(fused_clip)                       # [B, context_dim]
        ctx = ctx.unsqueeze(1).expand(-1, self.num_tokens, -1)  # [B, 77, context_dim]
        ctx = ctx + self.token_embed                      # bedakan tiap token
        return ctx


if __name__ == "__main__":
    m = CLIPToUNetContext(clip_dim=768)
    x = torch.randn(4, 768)
    y = m(x)
    print("ctx:", tuple(y.shape))  # (4, 77, 768)
    n_param = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"trainable params: {n_param/1e6:.3f}M")
