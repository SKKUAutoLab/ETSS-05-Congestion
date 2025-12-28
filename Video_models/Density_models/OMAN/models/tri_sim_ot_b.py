import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss

def similarity_cost(x1, x2, gamma=10):
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)
    sim = torch.bmm(x1, x2.transpose(1, 2)) * gamma
    sim = sim.exp()
    sim_c = sim.sum(dim=1, keepdim=True)
    sim_r = sim.sum(dim=2, keepdim=True)
    sim = sim / (sim_c + sim_r - sim)
    return 1 - sim

class GML(nn.Module):
    def __init__(self):
        super().__init__()
        self.ot = SamplesLoss(backend="tensorized", cost=similarity_cost, debias=False, diameter=3)
        self.margin=0.1
        self.dist = []

    def forward(self, x_pre, x_cur, dist):
        self.dist = dist.cuda()
        x1, x3 = x_pre
        x2, x4 = x_cur
        B = len(x1)
        N = [x.shape[0] for x in x1]
        M = [x.shape[0] for x in x2]
        N1 = [x.shape[0] for x in x3]
        M1 = [x.shape[0] for x in x4]
        device = x1[0].device
        ot_loss = []
        for b in range(B):
            assert N[b] == M[b]
            alpha = torch.cat((torch.ones(1, N[b], device=device), torch.zeros(1, N1[b], device=device)), dim=1)
            beta = torch.cat((torch.ones(1, N[b], device=device), torch.zeros(1, M1[b], device=device)), dim=1)
            f1 = torch.cat((x1[b], x3[b]), dim=0).unsqueeze(0)
            f2 = torch.cat((x2[b], x4[b]), dim=0).unsqueeze(0)
            loss = self.ot(alpha, f1, beta, f2)
            ot_loss.append(loss)
        loss=0
        loss += torch.relu(self.margin-torch.bmm(x3,x4.transpose(1,2))).sum()
        loss_dict = {"hinge_cost":loss.unsqueeze(0), "scon_cost":sum(ot_loss) / len(ot_loss)}
        return loss_dict