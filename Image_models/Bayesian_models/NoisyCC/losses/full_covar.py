from torch.nn.modules import Module
import torch

class Full_Cov_Gaussian_Loss(Module):
    def __init__(self, use_background, weight=0.01, reg=False):
        super(Full_Cov_Gaussian_Loss, self).__init__()
        self.use_bg = use_background
        self.w = weight
        self.reg = reg

    def forward(self, prob_list, target_list, pre_density):
        loss = 0
        B, C, H, W = pre_density.shape # [1, 1, 64, 64]
        for idx, prob in enumerate(prob_list):
            if prob is None:
                loss += torch.abs(0 - pre_density[idx].sum())
            else:
                N = len(target_list[idx]) 
                m = prob[0] 
                v = prob[1]
                B = prob[3]
                Minds = prob[4]
                if self.use_bg:
                    ann = torch.ones_like(m.sum(1)) # [8]
                    ann[:-1] = target_list[idx]
                    ann[-1] = 0
                    m = (ann.reshape(-1, 1) * m).sum(0).view(1, -1) # [1, 4096]
                else:
                    ann = target_list[idx]
                    m = (ann.reshape(-1,1)*m).sum(0).view(1,-1)
                factor = N / (m.sum() + 1e-12) # [1]
                m = m * factor # [1, 4096]
                v = v * (factor)**2 # [4096]
                B = B / (factor)**2 # [100, 100]
                x = pre_density[idx].reshape(1, C, H, W) # [1, 1, 64, 64]
                m = m.reshape(1, 1, H, W) # [1, 1, 64, 64]
                tmp = x - m # [1, 1, 64, 64]
                tmp = tmp.reshape(C, -1) # [1, 4096]
                lg1 = torch.sum(tmp**2 / v) # [1]
                lg2 = torch.sum(torch.mm(tmp[:, Minds], B) * tmp[:, Minds]) # [1]
                loss += (0.5 * (lg1 - lg2) * self.w)
                if self.reg:
                    p = prob[0]/ ((prob[0]).sum(0) + 1e-12) # [941, 4096]
                    pre_count = torch.sum(pre_density[idx].view(C, 1, -1) * p.unsqueeze(0), dim=2) # [1, 941]
                    pre_target = torch.ones_like((prob[0]).sum(1)) # [941]
                    if self.use_bg:
                        pre_target[:-1] = target_list[idx]
                        pre_target[-1] = 0
                    reg_loss = torch.sum(torch.abs(pre_count-pre_target.reshape(1, -1))) # [1]
                    loss += reg_loss
        loss = loss / len(prob_list)
        return loss