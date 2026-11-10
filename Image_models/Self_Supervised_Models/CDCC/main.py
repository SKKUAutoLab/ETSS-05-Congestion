import torch
from model import SFCN

if __name__ == "__main__":
    input = torch.rand(9, 3, 256, 256).cuda()
    model = SFCN().cuda()
    x, x_class = model(input) # [9, 1, 256, 256], [9, 2]
    print(x.shape)
    print(x_class.shape)
