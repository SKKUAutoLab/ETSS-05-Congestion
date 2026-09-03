#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP backbone — cabang global (paper §III-A, Eq. 1-2).

- ViT-L/14-336, OpenCLIP pretrained, di-FREEZE (paper: CLIP fully frozen).
- preprocess_pil(): TRAIN augment (CLAHE/flip/erase) ON, EVAL deterministic.
- forward(images, texts) -> (img_feat, txt_feat) ter-normalisasi.

Catatan open_clip v3: pakai get_tokenizer() (bukan open_clip.tokenize global),
karena API tokenizer berubah di major v3.
"""
import logging
import random
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

import open_clip
from PIL import Image

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


class CLIPBackbone(nn.Module):
    def __init__(
        self,
        model_name: str = "ViT-L-14-336",
        pretrained: str = "openai",
        device: str = "cuda",
        freeze: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logging.info(f"[CLIPBackbone] loading {model_name} ({pretrained})")

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = model.to(self.device)
        self.preprocess = preprocess
        # open_clip v3: tokenizer via get_tokenizer
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # augmentasi tambahan (hanya saat train=True)
        self.aug_flip = T.RandomHorizontalFlip(p=0.5)
        self.aug_erase = T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3))

        # embed dim
        if hasattr(self.model, "visual") and hasattr(self.model.visual, "output_dim"):
            self.embed_dim = self.model.visual.output_dim
        elif hasattr(self.model, "text_projection"):
            proj = self.model.text_projection
            self.embed_dim = proj.shape[1] if proj.ndim == 2 else proj.shape[0]
        else:
            self.embed_dim = getattr(self.model, "embed_dim", 768)

        self.freeze = freeze
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        logging.info(f"[CLIPBackbone] embed_dim={self.embed_dim}, freeze={freeze}")

    # ------------------------------------------------------
    def _apply_clahe_pil(self, img: Image.Image):
        if not _HAS_CV2:
            return img
        arr = np.array(img)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return img
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        rgb = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb)

    def preprocess_pil(self, images_pil: List[Image.Image], train: bool = True):
        tensors = []
        for img in images_pil:
            if train and _HAS_CV2 and random.random() < 0.5:
                img = self._apply_clahe_pil(img)
            tensors.append(self.preprocess(img))
        batch = torch.stack(tensors, dim=0)
        if train:
            batch = self.aug_flip(batch)
            batch = self.aug_erase(batch)
        return batch.to(self.device)

    # ------------------------------------------------------
    def _encode_image(self, images: torch.Tensor):
        images = images.to(self.device)
        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            feats = self.model.encode_image(images)
        return feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    def _encode_text(self, texts: Union[List[str], torch.Tensor]):
        if isinstance(texts, (list, tuple)):
            tokens = self.tokenizer(texts).to(self.device)
        else:
            tokens = texts.to(self.device)
        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            feats = self.model.encode_text(tokens)
        return feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    def forward(self, images, texts):
        return self._encode_image(images), self._encode_text(texts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("CLIPBackbone sanity (perlu unduh CLIP)...")
    try:
        bb = CLIPBackbone(device="cpu")
        imgs = [Image.new("RGB", (336, 336)) for _ in range(2)]
        x = bb.preprocess_pil(imgs, train=False)
        i, t = bb(x, ["crowd of people", "crowd of people"])
        print("img", tuple(i.shape), "txt", tuple(t.shape))
    except Exception as e:  # noqa
        print("[skip]", e)
