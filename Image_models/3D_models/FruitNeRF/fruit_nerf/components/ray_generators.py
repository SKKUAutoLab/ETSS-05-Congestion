from torch import Tensor, nn
import torch
from nerfstudio.cameras.rays import RayBundle

class OrthographicRayGenerator(nn.Module):
    image_coords: Tensor

    def __init__(self, surface_points, plane_normal, ray_batch_size, device, aabb) -> None:
        super().__init__()
        self.surface_points = surface_points
        self.surface_normal = torch.nn.functional.normalize(plane_normal).to(device)
        self.surface_vector_norm = torch.linalg.norm(plane_normal).to(device)
        self.ray_batch_size = ray_batch_size
        self.device = device
        self.aabb = aabb

    def forward(self, count) -> RayBundle:
        start = self.ray_batch_size * (count - 1)
        end = self.ray_batch_size * count
        if self.ray_batch_size * count >= self.surface_points.shape[0]:
            end = self.surface_points.shape[0]
        num_points = self.surface_points[start:end].shape[0]
        ray_bundle = RayBundle(origins=self.surface_points[start:end], directions=self.surface_normal.repeat(num_points, 1).to(self.device),
                               pixel_area=torch.zeros(num_points, 1).to(self.device), nears=torch.zeros(num_points, 1).to(self.device),
                               fars=torch.ones(num_points, 1).to(self.device) * self.surface_vector_norm)
        return ray_bundle