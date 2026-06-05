from torch.nn.modules import Module
import torch
import torch.nn.functional as F

class Bay_Loss(Module):
    def __init__(self, use_background):
        super(Bay_Loss, self).__init__()
        self.use_bg = use_background

    def forward(self, prob_list, target_list, pre_density):
        loss = 0
        for idx, prob in enumerate(prob_list):
            if prob is None:
                pre_count = torch.sum(pre_density[idx])
                target = torch.zeros((1,), dtype=torch.float32).cuda()
            else:
                N = len(prob)
                if self.use_bg:
                    target = torch.zeros((N,), dtype=torch.float32).cuda()
                    target[:-1] = target_list[idx]
                else:
                    target = target_list[idx]
                downsample_pre_density = F.interpolate(pre_density[idx].unsqueeze(0), scale_factor=0.5, mode='bilinear').squeeze(0)
                pre_count = torch.sum(downsample_pre_density.view((1, -1)) * prob, dim=1)
                # pre_count = torch.sum(pre_density[idx].view((1, -1)) * prob, dim=1)
            loss += torch.sum(torch.abs(target * 10 - pre_count))
        loss = loss / len(prob_list)
        return loss