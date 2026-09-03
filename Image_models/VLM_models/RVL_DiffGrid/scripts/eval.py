#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Eval RVL-DiffGrid (paper-aligned).

PERBEDAAN UTAMA vs eval lama:
  - eval LAMA menghitung MAE/RMSE dari cabang CLIP global (MLP head skalar).
    Padahal paper menyatakan global head DIBUANG saat inference.
  - eval BARU: count = spasial-integrasi density map dari cabang diffusion (Eq.16):
        C_hat = sum_{x,y} D_hat(x,y)
    CLIP + fusion(phi) tetap aktif HANYA utk menyuplai context cross-attention.

Kontrak forward model (rvlc.models.rvl_diffgrid.RVLDiffGrid):
    model.predict_density(images_pil, texts) -> density [B,1,H,W]
  (global counting head TIDAK dipanggil di sini.)

CATATAN: file ini bergantung pada modul yang masih perlu dibuat:
    rvlc/models/rvl_diffgrid.py   (wiring: CLIP -> ctx -> diffusion -> HSG/Phi -> Gamma)
    rvlc/models/hsg.py, refinement.py, density_regressor.py
Sampai modul itu ada, eval ini belum runnable end-to-end (sengaja eksplisit,
bukan stub yang berpura-pura jalan).
"""
import os
import sys
import json
import argparse
import logging

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from rvlc.data.dataset import CountDataset, collate_fn
from rvlc.models.rvl_diffgrid import RVLDiffGrid  # TODO: implement (wiring penuh)


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    abs_errors, sq_errors = [], []

    for batch in tqdm(dataloader, desc="[Eval]"):
        images_pil = batch["images_pil"]
        texts = batch["texts"]
        gt = batch["gt"].to(device)

        # ---- cabang diffusion: density map -> count (Eq.16) ----
        density = model.predict_density(images_pil, texts)   # [B,1,H,W]
        pred = density.flatten(1).sum(dim=1)                 # integrasi spasial
        pred = torch.clamp(pred, min=0.0).view_as(gt)

        err = torch.abs(pred - gt)
        abs_errors.append(err.cpu().numpy())
        sq_errors.append((err ** 2).cpu().numpy())

    abs_errors = np.concatenate(abs_errors) if abs_errors else np.array([np.inf])
    sq_errors = np.concatenate(sq_errors) if sq_errors else np.array([np.inf])

    mae = float(abs_errors.mean())
    rmse = float(np.sqrt(sq_errors.mean()))
    return mae, rmse


def parse_args():
    p = argparse.ArgumentParser(description="Eval RVL-DiffGrid (density-integration)")
    p.add_argument("--test_json", type=str, required=True)
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--batch_size", type=int, default=1)   # paper eval bs=1
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--clip_model", type=str, default="ViT-L-14-336")
    p.add_argument("--clip_pretrained", type=str, default="openai")
    p.add_argument("--sd_model", type=str, default="runwayml/stable-diffusion-v1-5")
    return p.parse_args()


def main(args):
    setup_logger()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device} | ckpt: {args.ckpt_path}")

    test_img_root = os.path.abspath(os.path.join(os.path.dirname(args.test_json), ".."))
    test_ds = CountDataset(args.test_json, img_root=test_img_root)
    if len(test_ds) == 0:
        raise RuntimeError("[FATAL] Test dataset kosong setelah resolusi path.")

    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn,
    )

    model = RVLDiffGrid(
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        sd_model=args.sd_model,
        device=str(device),
    ).to(device)

    # ---- load checkpoint (hanya bagian trainable; SD/CLIP frozen) ----
    state = torch.load(args.ckpt_path, map_location=device)
    sd = state.get("model_state_dict", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logging.info(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")

    mae, rmse = evaluate(model, test_loader, device)
    logging.info(f"[FINAL EVAL] MAE={mae:.2f}, RMSE={rmse:.2f}")


if __name__ == "__main__":
    main(parse_args())
