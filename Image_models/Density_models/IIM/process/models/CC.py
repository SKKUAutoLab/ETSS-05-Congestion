import torch
import torch.nn as nn
from . import counters

class CrowdCounter(nn.Module):
    def __init__(self,gpus,model_name):
        super(CrowdCounter, self).__init__()
        ccnet =  getattr(getattr(counters, model_name), model_name)
        self.CCN = ccnet()
        if len(gpus) > 1:
            self.CCN = torch.nn.DataParallel(self.CCN).cuda()
        else:
            self.CCN = self.CCN.cuda()
        self.loss_mse_fn = nn.MSELoss().cuda()
        
    @property
    def loss(self):
        return self.loss_mse
    
    def forward(self, img, dot_map):
        density_map = self.CCN(img)
        self.loss_mse= self.build_loss(density_map.squeeze(), dot_map.squeeze())
        return density_map, dot_map
    
    def build_loss(self, density_map, gt_data):
        loss_mse = self.loss_mse_fn(density_map, gt_data)  
        return loss_mse

    def test_forward(self, img):
        density_map = self.CCN(img)
        return density_map