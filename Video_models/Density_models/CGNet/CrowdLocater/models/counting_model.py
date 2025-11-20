import os
import torch
from torch import nn
from .backbone import build_backbone
from .counting_head import build_counting_head
from .utils import module2model
from torch.nn import functional as F
import numpy as np

class SingleScaleEncoderDecoderCounting(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.backbone = build_backbone(args.backbone)
        self.decoder_layers = build_counting_head(args.counting_head)
        self.stride = 4

    @torch.no_grad()
    def forward_points(self, x, loc_kernel_size=3): # [1, 3, 768, 1280]
        assert loc_kernel_size % 2 == 1
        assert x.shape[0] == 1
        z = self.backbone(x)
        out_dict = self.decoder_layers(z)
        predict_counting_map = out_dict["predict_counting_map"].detach().float() # [1, 1, 192, 320]
        pred_points = self._map_to_points(predict_counting_map,loc_kernel_size=loc_kernel_size,device=x.device)
        return [pred_points], predict_counting_map

    @torch.no_grad()
    def _map_to_points(self, predict_counting_map, loc_kernel_size=3, device="cuda"):
        loc_padding = loc_kernel_size // 2
        kernel = torch.ones(1,1,loc_kernel_size, loc_kernel_size).to(device).float()
        threshold = 0.5
        low_resolution_map = F.interpolate(F.relu(predict_counting_map), scale_factor=1)
        H, W = low_resolution_map.shape[-2], low_resolution_map.shape[-1]
        unfolded_map = F.unfold(low_resolution_map, kernel_size=loc_kernel_size, padding=loc_padding)
        unfolded_max_idx = unfolded_map.max(dim=1,keepdim=True)[1]
        unfolded_max_mask = (unfolded_max_idx==loc_kernel_size**2 // 2).reshape(1, 1, H, W)
        predict_cnt = F.conv2d(low_resolution_map,kernel, padding=loc_padding)
        predict_filter = (predict_cnt > threshold).float()
        predict_filter = predict_filter * unfolded_max_mask
        predict_filter = predict_filter.detach().cpu().numpy().astype(bool).reshape(H, W)
        pred_coord_weight = F.normalize(unfolded_map, p=1, dim=1)
        coord_h = torch.arange(H).reshape(-1, 1).repeat(1, W).to(device).float()
        coord_w = torch.arange(W).reshape(1, -1).repeat(H, 1).to(device).float()
        coord_h = coord_h.unsqueeze(0).unsqueeze(0)
        coord_w = coord_w.unsqueeze(0).unsqueeze(0)
        unfolded_coord_h = F.unfold(coord_h, kernel_size=loc_kernel_size, padding=loc_padding)
        pred_coord_h = (unfolded_coord_h * pred_coord_weight).sum(dim=1, keepdim=True).reshape(H,W).detach().cpu().numpy()
        unfolded_coord_w = F.unfold(coord_w, kernel_size=loc_kernel_size, padding=loc_padding)
        pred_coord_w = (unfolded_coord_w * pred_coord_weight).sum(dim=1, keepdim=True).reshape(H, W).detach().cpu().numpy()
        coord_h = pred_coord_h[predict_filter].reshape(-1, 1)
        coord_w = pred_coord_w[predict_filter].reshape(-1, 1)
        coord = np.concatenate([coord_w, coord_h], axis=1)
        pred_points = [[self.stride * coord_w + 0.5, self.stride * coord_h + 0.5] for coord_w,coord_h in coord]
        return pred_points

def build_counting_model(args):
    model = SingleScaleEncoderDecoderCounting(args)
    if os.path.exists(args.ckpt_dir):
        print("load ckpt from", args.ckpt_dir)
        ckpt = torch.load(args.ckpt_dir, map_location="cpu")
        state_dict = module2model(ckpt['model'])
        model.load_state_dict(state_dict, strict=False)
    return model