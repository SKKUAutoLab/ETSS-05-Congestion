import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, normalize=False):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, patch_embedding, gt_text_embedding_map, gt_density): # [8, 196, 512], [8, 1, 512], [8, 384, 384]
        gt_density = F.interpolate(gt_density.unsqueeze_(1), size=(224, 224), mode='nearest')
        density_mask = F.max_pool2d(gt_density, kernel_size=16, stride=16, padding=0)
        density_mask = density_mask > 0.
        density_mask = density_mask.permute(0, 2, 3 ,1)
        gt_text_embedding_map = gt_text_embedding_map.unsqueeze(1).expand(-1, 14, 14, -1)
        fused_text_embedding_map =  gt_text_embedding_map
        pos_mask = density_mask.squeeze_(-1)
        patch_embeddings = patch_embedding.reshape(-1, 14, 14, 512)
        sim_map = F.cosine_similarity(patch_embeddings, fused_text_embedding_map , dim=-1)
        n_pos = torch.sum(pos_mask, dim=(1, 2))
        n_pos = torch.where(n_pos == 0, torch.ones_like(n_pos), n_pos)
        sim_map = torch.exp(sim_map / self.temperature)
        pos_sum = torch.sum(torch.where(pos_mask, sim_map, torch.zeros_like(sim_map)), dim=(1, 2)) + 1e-5
        neg_sum = torch.sum(torch.where(~pos_mask, sim_map, torch.zeros_like(sim_map)), dim=(1, 2)) + 1e-5
        loss = -torch.log(pos_sum / (pos_sum + neg_sum))
        if self.normalize:
            loss = loss / n_pos
        return loss.mean()