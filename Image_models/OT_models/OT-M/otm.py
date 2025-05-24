import torch
import torch.nn.functional as tF
from simple_ot import SampleOT
import cv2
import matplotlib.pyplot as plt

eps = 1e-12

class L2_DIS:
    factor = 1 / 32

    @staticmethod
    def __call__(X, Y): # [1, 843, 2], [1, 190, 2]
        x_col = X.unsqueeze(-2) # [1, 843, 1, 2]
        y_row = Y.unsqueeze(-3) # [1, 1, 190, 2]
        C = ((x_col - y_row) ** 2).sum(dim=-1) / 2 # [1, 843, 190]
        return C * L2_DIS.factor

    @staticmethod
    def barycenter(weight, coord): # [1, 843, 190], [1, 843, 2]
        weight = weight / (weight.sum(dim=1, keepdim=True) + eps) # [1, 843, 190]
        return weight.permute(0, 2, 1) @ coord

blur = 0.01
per_cost = L2_DIS()
ot = SampleOT(blur=blur, scaling=0.9, reach=None, fixed_epsilon=False)

def den2coord(denmap, scale_factor=8):
    coord = torch.nonzero(denmap > eps) # [843, 2]
    denval = denmap[coord[:, 0], coord[:, 1]] # [843]
    if scale_factor != 1:
        coord = coord.float() * scale_factor + scale_factor / 2 # [843, 2]
    return denval.reshape(1, -1, 1), coord.reshape(1, -1, 2)

def init_dot(denmap, n, scale_factor=8):
    norm_den = denmap[None, None, ...] # [1, 1, 96, 128]
    norm_den = tF.interpolate(norm_den, scale_factor=scale_factor, mode='bilinear', align_corners=False) # [1, 1, 768, 1024]
    norm_den = norm_den[0, 0] # [768, 1024]
    d_coord = torch.nonzero(norm_den > eps) # [103808, 2]
    norm_den = norm_den[d_coord[:, 0], d_coord[:, 1]] # [103808]
    cidx = torch.multinomial(norm_den, num_samples=n, replacement=False) # [190]
    coord = d_coord[cidx] # [190, 2]
    B = torch.ones(1, n, 1).to(denmap) # [1, 190, 1]
    B_coord = coord.reshape(1, n, 2) # [1, 190, 2]
    return B, B_coord

@torch.no_grad()
def OT_M(A, A_coord, B, B_coord, scale_factor=8, max_itern=8):
    for iter in range(max_itern):
        C = per_cost(A_coord, B_coord) # [1, 843, 190]
        F, G = ot(A, B, C) # [1, 843, 1], [1, 190, 1]
        PI = ot.plan(A, B, F, G, C) # [1, 843, 190]
        nB_coord = per_cost.barycenter(PI, A_coord) # [1, 190, 2]
        move = torch.norm(nB_coord - B_coord, p=2, dim=-1) # [1, 190]
        if move.mean().item() < 1 and move.max().item() < scale_factor:
            break
        B_coord = nB_coord # [1, 190, 2]
    return (nB_coord).reshape(-1, 2)

@torch.no_grad()
def den2seq(denmap, scale_factor=8, max_itern=16, ot_scaling=0.75):
    ot.scaling = ot_scaling
    assert denmap.dim() == 2, f"the shape of density map should be [H, W], but the given one is {denmap.shape}"
    num = int(denmap.sum().item() + 0.5)
    if num < 0.5:
        return torch.zeros((0, 2)).to(denmap)
    denmap = denmap * num / denmap.sum() # [96, 128]
    A, A_coord = den2coord(denmap, scale_factor) # [1, 843, 1], [1, 843, 2]
    B, B_coord = init_dot(denmap, num, scale_factor) # [1, 190, 1], [1, 190, 2]
    flocs = OT_M(A, A_coord, B, B_coord, scale_factor, max_itern=max_itern) # [190, 2]
    return flocs

@torch.no_grad()
def main():
    img = cv2.imread('samples/real.jpg') # [768, 1024, 3]
    imh, imw = img.shape[:2]
    denmap = torch.load('samples/real.pth') # [96, 128]
    dh, dw = denmap.shape
    scale_factor = imw / dw
    plt.imsave('denmap.png', denmap.cpu(), cmap='jet')
    dot = den2seq(denmap, scale_factor) # [190, 2]
    dot_coord = dot.long().cpu()
    dotmap = torch.zeros((imh, imw)) # [768, 1024]
    dotmap[dot_coord[:, 0], dot_coord[:, 1]] = 1
    dotmap = tF.conv2d(dotmap[None, None, ...], torch.ones((1, 1, 5, 5)), padding=2)[0, 0] # [768, 1024]
    plt.imsave("dotmap.png", dotmap)
    
if __name__ == '__main__':
    main()