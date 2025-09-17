import torch.nn as nn
import torch

class RATEnet(nn.Module):
    def __init__(self):
        super(RATEnet, self).__init__()
        self.des_dimension = nn.Sequential(nn.Conv2d(152,64,3,padding=1), nn.ReLU(inplace=True))
        self.ROI_feat = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True), nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                      nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True), nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
        self.output = nn.Sequential(nn.Linear(32 * 3 * 3, 1))
        self._initialize_weights()

    def forward(self, x):
        x = self.des_dimension(x)
        x = self.ROI_feat(x)
        x = x.view(x.size(0), 32 * 3 * 3)
        x = torch.abs(self.output(x))
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)