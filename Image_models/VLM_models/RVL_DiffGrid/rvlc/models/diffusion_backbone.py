#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RVL-DiffGrid — Diffusion-Guided Latent Encoding & Cross-Attention Extraction.

Sesuai paper §III-C:
  - VAE encode image (resize 512x512) -> z0 [B,4,64,64]
  - Single forward diffusion step di timestep TETAP t=500 (Eq.8)
  - Frozen text-conditioned U-Net (Eq.9), context = fused CLIP embedding
    yang sudah diproyeksikan ke 768-dim & di-reshape ke N=77 token
    (lihat clip_to_unet_context.CLIPToUNetContext). Text encoder SD DI-BYPASS.
  - Ekstrak cross-attention maps di tiga skala {64,32,16} (Eq.11)
    + multi-scale latent feature F = {F64,F32,F16} (Eq.10) dari decoder U-Net.

PENTING (perbaikan dari versi lama):
  Versi lama memasang forward-hook pada modul `attn2` dan mengasumsikan output
  4D [B,heads,tokens,spatial]. Di diffusers, output `attn2` adalah attended
  hidden states 3D [B, seq, dim] -> map attention TIDAK pernah tertangkap
  (semua ter-skip oleh `if attn.dim()!=4: continue`), sehingga cabang diffusion
  diisi dummy zeros. Di sini kita pakai custom AttnProcessor yang menghitung
  `attn.get_attention_scores(...)` dan menyimpan probabilitas attention asli.

Diuji terhadap API diffusers yang mengekspos Attention.get_attention_scores /
head_to_batch_dim / batch_to_head_dim (stabil dari ~0.21 s/d main saat ini).
"""

import logging
import math
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from diffusers import StableDiffusionPipeline


# ==========================================================
# Custom Attention Processor: simpan cross-attention probs asli
# ==========================================================
class CrossAttnStoreProcessor:
    """
    Meniru AttnProcessor default diffusers, tapi menyimpan attention_probs
    untuk layer cross-attention (attn2) ke dalam `store`.

    store : dict[str, dict]  ->  store[name] = {"probs": [B*heads,q,kv], "heads": int}
    Hanya menyimpan saat cross-attention (encoder_hidden_states is not None).
    """

    def __init__(self, store: Dict[str, dict], name: str):
        self.store = store
        self.name = name

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        **kwargs,
    ):
        residual = hidden_states

        # input bisa 4D [B,C,H,W] (jarang utk SD1.5 attn) -> ratakan ke 3D
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        is_cross = encoder_hidden_states is not None

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # probs: [B*heads, q_tokens, kv_tokens]
        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # simpan HANYA cross-attention (attn2)
        if is_cross:
            self.store[self.name] = {
                "probs": attention_probs.detach(),
                "heads": attn.heads,
            }

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj + dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(1, 2).view(b, c, h, w)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# ==========================================================
# Diffusion backbone
# ==========================================================
class RVLDiffusionBackbone(nn.Module):
    """
    Frozen Stable Diffusion (default v1.5) sebagai prior spasial.

    extract(images_pil, context_emb) -> dict:
        "attn": {64: [B,1,64,64], 32: [B,1,32,32], 16: [B,1,16,16]}
        "feat": {64: [B,C,64,64], 32: [B,C,32,32], 16: [B,C,16,16]}
    """

    # skala spasial yang dipakai paper (Eq.10/Eq.11)
    TARGET_SCALES = (64, 32, 16)

    def __init__(
        self,
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        fixed_timestep: int = 500,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.fixed_timestep = fixed_timestep

        # dtype: fp16 di GPU, fp32 di CPU (kecuali dioverride)
        if dtype is None:
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.dtype = dtype

        logging.info(
            f"[Diff] Loading StableDiffusionPipeline: {pretrained_model_name_or_path}"
        )
        pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=self.dtype
        ).to(self.device)

        self.unet = pipe.unet
        self.vae = pipe.vae
        self.scheduler = pipe.scheduler
        # NOTE: pipe.text_encoder / pipe.tokenizer SENGAJA tidak dipakai.
        # Sesuai paper, context cross-attention berasal dari embedding CLIP
        # yang diproyeksikan (bypass text encoder SD).

        # freeze semua (kita tidak fine-tune SD)
        for m in (self.unet, self.vae):
            for p in m.parameters():
                p.requires_grad = False
            m.eval()

        # storage attn + hooks fitur decoder
        self._attn_store: Dict[str, dict] = {}
        self._feat_store: Dict[str, torch.Tensor] = {}
        self._install_attn_processors()
        self._register_decoder_feature_hooks()

        # preprocess image -> 512x512, [-1,1]
        self.img_transform = T.Compose(
            [
                T.Resize((512, 512), interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    # ------------------------------------------------------
    # Pasang custom processor di SEMUA cross-attention (attn2)
    # ------------------------------------------------------
    def _install_attn_processors(self):
        procs = {}
        for name in self.unet.attn_processors.keys():
            # nama cross-attn diffusers berakhiran "attn2.processor"
            if name.endswith("attn2.processor"):
                procs[name] = CrossAttnStoreProcessor(self._attn_store, name)
            else:
                # pertahankan processor default utk self-attn (attn1)
                procs[name] = self.unet.attn_processors[name]
        self.unet.set_attn_processor(procs)
        n_cross = sum(1 for k in procs if k.endswith("attn2.processor"))
        logging.info(f"[Diff] Installed CrossAttnStoreProcessor on {n_cross} attn2 layers.")

    # ------------------------------------------------------
    # Hook fitur decoder U-Net (output up_blocks) -> [B,C,H,W]
    # ------------------------------------------------------
    def _register_decoder_feature_hooks(self):
        def make_hook(idx):
            def hook(module, inp, out):
                # output up_block bisa berupa tensor atau tuple
                feat = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(feat) and feat.ndim == 4:
                    self._feat_store[f"up_{idx}"] = feat
            return hook

        for i, blk in enumerate(self.unet.up_blocks):
            blk.register_forward_hook(make_hook(i))
        logging.info(f"[Diff] Registered decoder hooks on {len(self.unet.up_blocks)} up_blocks.")

    # ------------------------------------------------------
    @torch.no_grad()
    def _encode_to_latent(self, images_pil: List[Image.Image]) -> torch.Tensor:
        imgs = torch.stack([self.img_transform(im) for im in images_pil], dim=0)
        imgs = imgs.to(self.device, dtype=self.vae.dtype)
        latents = self.vae.encode(imgs).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor  # 0.18215 utk SD1.5
        return latents

    # ------------------------------------------------------
    # Ubah probs [B*heads,q,kv] -> peta spasial [B,1,s,s]
    #   (rata-rata over heads & over semua text token, Eq.11)
    # ------------------------------------------------------
    @staticmethod
    def _probs_to_spatial(probs: torch.Tensor, heads: int, batch: int) -> Optional[torch.Tensor]:
        bh, q, kv = probs.shape
        if bh != batch * heads:
            return None
        s = int(round(math.sqrt(q)))
        if s * s != q:
            return None  # bukan peta persegi (skip)
        a = probs.view(batch, heads, q, kv)
        a = a.mean(dim=1)        # rata-rata head -> [B,q,kv]
        a = a.mean(dim=-1)       # agregasi semua token teks -> [B,q]
        a = a.view(batch, 1, s, s).float()
        return a

    # ------------------------------------------------------
    @torch.no_grad()
    def extract(self, images_pil: List[Image.Image], context_emb: torch.Tensor) -> Dict[str, Dict[int, torch.Tensor]]:
        """
        images_pil : list PIL Image
        context_emb: [B, 77, 768] hasil proyeksi CLIP (lihat CLIPToUNetContext)

        return: {"attn": {s: [B,1,s,s]}, "feat": {s: [B,C,s,s]}} utk s in {64,32,16}
        """
        B = len(images_pil)
        self._attn_store.clear()
        self._feat_store.clear()

        latents = self._encode_to_latent(images_pil)  # [B,4,64,64]
        context_emb = context_emb.to(self.device, dtype=self.unet.dtype)

        # single forward step di t TETAP = 500 (Eq.8)
        t = torch.full((B,), self.fixed_timestep, device=self.device, dtype=torch.long)
        noise = torch.randn_like(latents)
        noisy = self.scheduler.add_noise(latents, noise, t)

        # satu kali UNet (tidak ada denoising loop, tidak ada grad ke SD)
        _ = self.unet(noisy, t, encoder_hidden_states=context_emb)

        # ---- kumpulkan cross-attn per skala ----
        per_scale_maps: Dict[int, List[torch.Tensor]] = {s: [] for s in self.TARGET_SCALES}
        for name, rec in self._attn_store.items():
            sp = self._probs_to_spatial(rec["probs"], rec["heads"], B)
            if sp is None:
                continue
            s = sp.shape[-1]
            if s in per_scale_maps:
                per_scale_maps[s].append(sp)

        attn_out: Dict[int, torch.Tensor] = {}
        for s, maps in per_scale_maps.items():
            if len(maps) == 0:
                # tidak ada layer di skala ini -> isi nol (jarang utk SD1.5)
                attn_out[s] = torch.zeros(B, 1, s, s, device=self.device)
            else:
                attn_out[s] = torch.stack(maps, dim=0).mean(0)  # rata-rata antar-layer

        # ---- kumpulkan fitur decoder per skala ----
        feat_by_res: Dict[int, torch.Tensor] = {}
        for _, feat in self._feat_store.items():
            res = feat.shape[-1]
            # ambil yang resolusinya tepat di target; kalau ada beberapa, pakai terakhir
            if res in self.TARGET_SCALES:
                feat_by_res[res] = feat.float()

        feat_out: Dict[int, torch.Tensor] = {}
        for s in self.TARGET_SCALES:
            if s in feat_by_res:
                feat_out[s] = feat_by_res[s]
            else:
                # fallback: interpolasi dari resolusi terdekat yang ada
                if feat_by_res:
                    nearest = min(feat_by_res.keys(), key=lambda r: abs(r - s))
                    feat_out[s] = F.interpolate(
                        feat_by_res[nearest], size=(s, s),
                        mode="bilinear", align_corners=False,
                    )
                else:
                    feat_out[s] = torch.zeros(B, 1, s, s, device=self.device)

        return {"attn": attn_out, "feat": feat_out}

    def forward(self, images_pil, context_emb):
        return self.extract(images_pil, context_emb)


# ==========================================================
# Sanity check (butuh download SD; berat — jalankan manual)
# ==========================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Sanity check RVLDiffusionBackbone (perlu koneksi + VRAM)...")
    try:
        bb = RVLDiffusionBackbone(device="cuda")
        dummy_imgs = [Image.new("RGB", (512, 512), (128, 128, 128)) for _ in range(2)]
        ctx = torch.randn(2, 77, 768)
        out = bb.extract(dummy_imgs, ctx)
        for s in bb.TARGET_SCALES:
            print(f"  scale {s}: attn={tuple(out['attn'][s].shape)}, feat={tuple(out['feat'][s].shape)}")
    except Exception as e:  # noqa
        print(f"[skip] {type(e).__name__}: {e}")
