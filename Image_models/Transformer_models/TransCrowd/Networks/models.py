import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

class VisionTransformer_token(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        trunc_normal_(self.pos_embed, std=.02)
        self.output1 = nn.Sequential(nn.ReLU(), nn.Dropout(0.5), nn.Linear(1000, 1))
        self.output1.apply(self._init_weights)

    def forward_features(self, x): # [8, 3, 384, 384]
        B = x.shape[0]
        x = self.patch_embed(x) # [8, 576, 768]
        cls_tokens = self.cls_token.expand(B, -1, -1) # [8, 1, 768]
        x = torch.cat((cls_tokens, x), dim=1) # [8, 577, 768]
        x = x + self.pos_embed # [8, 577, 768]
        x = self.pos_drop(x) # [8, 577, 768]
        for blk in self.blocks:
            x = blk(x) # [8, 577, 768]
        x = self.norm(x) # [8, 577, 768]
        return x[:, 0]

    def forward(self, x): # [8, 3, 384, 384]
        x = self.forward_features(x) # [8, 768]
        x = self.head(x) # [8, 1000]
        x = self.output1(x) # [8, 1]
        return x

class VisionTransformer_gap(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        trunc_normal_(self.pos_embed, std=.02)
        self.output1 = nn.Sequential(nn.ReLU(), nn.Linear(6912 * 4, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 1))
        self.output1.apply(self._init_weights)

    def forward_features(self, x): # [8, 3, 384, 384]
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x[:, 1:] # [8, 576, 768]
        return x

    def forward(self, x): # [8, 3, 384, 384]
        x = self.forward_features(x) # [8, 576, 768]
        x = F.adaptive_avg_pool1d(x, (48)) # [8, 576, 48]
        x = x.view(x.shape[0], -1) # [8, 27648]
        x = self.output1(x) # [8, 1]
        return x

@register_model
def base_patch16_384_token(pretrained=False, **kwargs):
    model = VisionTransformer_token(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load('Networks/deit_base_patch16_384-8de9b5d1.pth')
        model.load_state_dict(checkpoint["model"], strict=False)
        print("load transformer ckpt from: Networks/deit_base_patch16_384-8de9b5d1.pth")
    return model

@register_model
def base_patch16_384_gap(pretrained=False, **kwargs):
    model = VisionTransformer_gap(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load('Networks/deit_base_patch16_384-8de9b5d1.pth')
        model.load_state_dict(checkpoint["model"], strict=False)
        print("Load transformer ckpt from: 'Networks/deit_base_patch16_384-8de9b5d1.pth'")
    return model