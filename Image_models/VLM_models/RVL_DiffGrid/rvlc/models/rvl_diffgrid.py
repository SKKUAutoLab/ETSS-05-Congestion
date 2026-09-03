#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RVL-DiffGrid — full dual-branch model (paper Figure 1).

Cabang global (training-only):  CLIP -> Fusion(phi) -> GlobalCountHead
Cabang diffusion (inference):    CLIP->phi -> CLIPToUNetContext -> frozen U-Net
                                 -> {attn, feat} -> Refinement(Phi) + HSG -> Gamma
Count akhir = integrasi density map (Eq.16).

Pembagian gradien (sesuai paper "no gradient into diffusion backbone"):
  - extract() di RVLDiffusionBackbone memakai no_grad + detach -> tidak ada
    gradien yang masuk ke VAE/U-Net, dan juga tidak ke CLIP->phi->ctx lewat U-Net.
  - Trainable: fusion, ctx, global_head (lewat L_global); refine, regressor
    (lewat L_dens / L_struct / bagian density dari L_ADR).
"""
from typing import Dict, List

import torch
import torch.nn as nn
from PIL import Image

from .clip_backbone import CLIPBackbone
from .fusion import Fusion
from .global_count_head import GlobalCountHead
from .clip_to_unet_context import CLIPToUNetContext
from .diffusion_backbone import RVLDiffusionBackbone
from .refinement import Refinement
from .hsg import HSG
from .density_regressor import DensityRegressor


# nama prefix modul TRAINABLE (untuk simpan/muat checkpoint ringkas)
TRAINABLE_PREFIXES = ("fusion.", "global_head.", "ctx.", "refine.", "regressor.")


class RVLDiffGrid(nn.Module):
    def __init__(
        self,
        clip_model: str = "ViT-L-14-336",
        clip_pretrained: str = "openai",
        sd_model: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        fusion_mode: str = "linear",
        fixed_timestep: int = 500,
        refine_dim: int = 256,
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # ---- frozen encoders ----
        self.clip = CLIPBackbone(clip_model, clip_pretrained, device=str(self.device), freeze=True)
        D = self.clip.embed_dim
        self.diff = RVLDiffusionBackbone(sd_model, device=str(self.device), fixed_timestep=fixed_timestep)

        # ---- trainable ----
        self.fusion = Fusion(dim=D, mode=fusion_mode)
        self.global_head = GlobalCountHead(in_dim=D)
        self.ctx = CLIPToUNetContext(clip_dim=D, context_dim=768, num_tokens=77)
        self.refine = Refinement(out_dim=refine_dim, target_scale=64)
        self.hsg = HSG(grid_size=16, scales=(64, 32, 16))
        self.regressor = DensityRegressor(in_dim=refine_dim)

    # ------------------------------------------------------
    def encode_fused(self, images_clip: torch.Tensor, texts: List[str]) -> torch.Tensor:
        img_feat, txt_feat = self.clip(images_clip, texts)
        return self.fusion(img_feat, txt_feat)  # [B, D]

    def _diffusion_density(self, images_pil: List[Image.Image], fused: torch.Tensor):
        context = self.ctx(fused)                      # [B,77,768]
        out: Dict[str, Dict[int, torch.Tensor]] = self.diff.extract(images_pil, context)
        fref = self.refine(out["feat"])                # [B,C,64,64]
        S = self.hsg(out["attn"])                      # [B,1,16,16]
        density = self.regressor(fref, S)              # [B,1,H,W]
        return density, S

    # ------------------------------------------------------
    @torch.no_grad()
    def predict_density(self, images_pil: List[Image.Image], texts: List[str]) -> torch.Tensor:
        """Jalur INFERENCE (Eq.16): global head dibuang."""
        self.eval()
        images_clip = self.clip.preprocess_pil(images_pil, train=False)
        fused = self.encode_fused(images_clip, texts)
        density, _ = self._diffusion_density(images_pil, fused)
        return density  # [B,1,H,W]; count = density.flatten(1).sum(1)

    def forward(self, images_pil: List[Image.Image], texts: List[str], train: bool = True) -> Dict[str, torch.Tensor]:
        """Jalur TRAINING: kembalikan semua yang dibutuhkan loss."""
        images_clip = self.clip.preprocess_pil(images_pil, train=train)
        fused = self.encode_fused(images_clip, texts)
        global_count = self.global_head(fused)         # [B]
        density, S = self._diffusion_density(images_pil, fused)
        return {
            "global_count": global_count,              # C_global (auxiliary)
            "density": density,                        # D_hat [B,1,H,W]
            "grid_S": S,                               # S [B,1,16,16]
            "count": self.regressor.integrate(density) # sum D_hat (Eq.16) [B]
        }

    # ------------------------------------------------------
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_state_dict(self):
        return {
            k: v for k, v in self.state_dict().items()
            if k.startswith(TRAINABLE_PREFIXES)
        }


if __name__ == "__main__":
    print("RVLDiffGrid wiring check (perlu unduh CLIP+SD; berat)...")
    try:
        m = RVLDiffGrid(device="cpu")
        imgs = [Image.new("RGB", (512, 512)) for _ in range(2)]
        out = m(imgs, ["crowd of people"] * 2)
        for k, v in out.items():
            print(f"  {k}: {tuple(v.shape)}")
        n = sum(p.numel() for p in m.trainable_parameters())
        print(f"  trainable params: {n/1e6:.3f}M")
    except Exception as e:  # noqa
        print("[skip]", e)
