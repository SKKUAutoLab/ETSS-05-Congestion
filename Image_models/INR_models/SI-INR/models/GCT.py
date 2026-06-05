import torch
from torch import nn

class GCT(nn.Module):
    def __init__(self, num_channels, k_size=3, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x): # [4, 512, 32, 32]
        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2,3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            embedding = self.conv(embedding.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2,3), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:
            print('This mode does not exist')
            raise NotImplementedError
        gate = 1. + torch.tanh(embedding * norm + self.beta)
        return x * gate # [4, 512, 32, 32]