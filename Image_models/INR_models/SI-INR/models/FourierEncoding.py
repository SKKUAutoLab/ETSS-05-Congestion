import torch
import numpy as np
import torch.nn as nn
from torch import Tensor

@torch.jit.script
def positional_encoding(v: Tensor, m: int) -> Tensor:
    j = torch.arange(m, device=v.device)
    coeffs = 2.0** j * np.pi
    vp = coeffs * torch.unsqueeze(v, -1)
    vp_cat = torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)
    return vp_cat.flatten(-2, -1)

class PositionalEncoding(nn.Module):
    def __init__(self, m: int):
        super().__init__()
        self.m = m

    def forward(self, v: Tensor) -> Tensor:
        return positional_encoding(v, self.m)