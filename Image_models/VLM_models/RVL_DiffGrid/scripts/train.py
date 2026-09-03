#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train RVL-DiffGrid (paper Eq. 20):

  L_total = L_dens + L_global + L_struct
          = L_dens
          + (L_Huber + lrel*L_rel + lADR*L_ADR)     # Eq.4-6
          + (lLRRC*L_LRRC + lgrid*L_grid)           # Eq.17-19

Hanya fusion/ctx/global_head/refine/regressor yang dilatih (CLIP & SD frozen).
Checkpoint disimpan ringkas (hanya bobot trainable).
"""
import os
import sys
import argparse
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    import yaml
except ImportError:
    yaml = None

from rvlc.data.dataset import CountDataset, collate_fn
from rvlc.data.density_map import build_density_map
from rvlc.models.rvl_diffgrid import RVLDiffGrid
from rvlc.losses import global_loss, density_loss, struct_loss
from rvlc.utils.metrics import mae_rmse


def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def build_gt_density(batch, out_size, mode, device):
    """Bangun GT density [B,1,Hg,Wg] dari points; None bila tidak ada points sama sekali."""
    maps = []
    any_pts = False
    for pts, (W, H) in zip(batch["points"], batch["orig_size"]):
        if pts is not None and len(pts) > 0:
            any_pts = True
        d = build_density_map(pts, (W, H), out_size=out_size, mode=mode)
        maps.append(torch.from_numpy(d))
    if not any_pts:
        return None
    return torch.stack(maps, dim=0).unsqueeze(1).to(device)  # [B,1,Hg,Wg]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    errs = []
    for batch in loader:
        gt = batch["gt"].to(device)
        density = model.predict_density(batch["images_pil"], batch["texts"])
        pred = density.flatten(1).sum(dim=1).clamp(min=0.0).view_as(gt)
        errs.append(torch.abs(pred - gt).cpu().numpy())
    return mae_rmse(np.concatenate(errs)) if errs else (float("inf"), float("inf"))


def parse_args():
    p = argparse.ArgumentParser(description="Train RVL-DiffGrid")
    p.add_argument("--train_json", type=str, required=True)
    p.add_argument("--test_json", type=str, default=None)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--config", type=str, default=None, help="YAML hyperparameter (Tabel V)")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--lambda_rel", type=float, default=0.03)
    p.add_argument("--lambda_adr", type=float, default=0.01)
    p.add_argument("--lambda_lrrc", type=float, default=0.05)
    p.add_argument("--lambda_grid", type=float, default=0.05)
    p.add_argument("--density_mode", type=str, default="adaptive", choices=["adaptive", "fixed"])
    p.add_argument("--out_size", type=int, default=64, help="resolusi density map (HxW)")
    p.add_argument("--clip_model", type=str, default="ViT-L-14-336")
    p.add_argument("--clip_pretrained", type=str, default="openai")
    p.add_argument("--sd_model", type=str, default="runwayml/stable-diffusion-v1-5")
    args = p.parse_args()

    # config yaml menimpa default (kalau ada)
    if args.config and yaml is not None and os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
        logging.info(f"[CONFIG] loaded {args.config}: {cfg}")
    return args


def main(args):
    setup_logger()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    out_size = (args.out_size, args.out_size)

    train_root = os.path.abspath(os.path.join(os.path.dirname(args.train_json), ".."))
    train_ds = CountDataset(args.train_json, img_root=train_root)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn,
    )
    test_loader = None
    if args.test_json:
        test_root = os.path.abspath(os.path.join(os.path.dirname(args.test_json), ".."))
        test_ds = CountDataset(args.test_json, img_root=test_root)
        test_loader = DataLoader(
            test_ds, batch_size=1, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn,
        )

    model = RVLDiffGrid(
        clip_model=args.clip_model, clip_pretrained=args.clip_pretrained,
        sd_model=args.sd_model, device=str(device),
    ).to(device)

    # Refinement uses LazyConv2d because diffusion feature channels are inferred
    # from the first U-Net pass. Build the optimizer after that first forward.
    optim = None

    best_mae = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {}
        for batch in train_loader:
            gt = batch["gt"].to(device)
            gt_density = build_gt_density(batch, out_size, args.density_mode, device)

            out = model(batch["images_pil"], batch["texts"], train=True)
            if optim is None:
                params = list(model.trainable_parameters())
                optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
                n_train = sum(p.numel() for p in params)
                logging.info(f"[MODEL] trainable params: {n_train/1e6:.3f}M")

            # --- L_dens (Eq.20, primary) ---
            l_dens = density_loss(out["density"], gt_density) if gt_density is not None \
                else torch.tensor(0.0, device=device)
            # --- L_global (Eq.4-6) ---
            l_glob, glog = global_loss(
                out["global_count"], out["count"], gt,
                lambda_rel=args.lambda_rel, lambda_adr=args.lambda_adr,
            )
            # --- L_struct (Eq.17-19) ---
            l_struct, slog = struct_loss(
                out["grid_S"], out["density"], gt_density,
                lambda_lrrc=args.lambda_lrrc, lambda_grid=args.lambda_grid,
            )

            loss = l_dens + l_glob + l_struct

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            optim.step()

            for k, v in {"loss": loss.detach(), "l_dens": l_dens.detach(), **glog, **slog}.items():
                running[k] = running.get(k, 0.0) + float(v)

        msg = " ".join(f"{k}={v/len(train_loader):.4f}" for k, v in running.items())
        logging.info(f"[Epoch {epoch}/{args.epochs}] {msg}")

        if test_loader is not None:
            mae, rmse = evaluate(model, test_loader, device)
            logging.info(f"[Eval epoch {epoch}] MAE={mae:.2f} RMSE={rmse:.2f}")
            if mae < best_mae:
                best_mae = mae
                ckpt = {"model_state_dict": model.trainable_state_dict(),
                        "epoch": epoch, "mae": mae, "rmse": rmse}
                torch.save(ckpt, os.path.join(args.ckpt_dir, "best_mae.pth"))
                logging.info(f"[CKPT] saved best_mae.pth (MAE={mae:.2f})")

    # simpan terakhir
    torch.save({"model_state_dict": model.trainable_state_dict(), "epoch": args.epochs},
               os.path.join(args.ckpt_dir, "last.pth"))
    logging.info("Training selesai.")


if __name__ == "__main__":
    main(parse_args())
