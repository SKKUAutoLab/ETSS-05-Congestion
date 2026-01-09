from typing import Optional
import torch
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.model_components.ray_samplers import SpacedSampler

class UniformSamplerWithNoise(SpacedSampler):
    def __init__(self, num_samples: Optional[int] = None, train_stratified=True, single_jitter=False) -> None:
        super().__init__(num_samples=num_samples, spacing_fn=lambda x: x, spacing_fn_inv=lambda x: x, train_stratified=train_stratified, single_jitter=single_jitter)

    def generate_ray_samples(self, ray_bundle: Optional[RayBundle] = None, num_samples: Optional[int] = None) -> RaySamples:
        assert ray_bundle is not None
        assert ray_bundle.nears is not None
        assert ray_bundle.fars is not None
        num_samples = num_samples or self.num_samples
        assert num_samples is not None
        num_rays = ray_bundle.origins.shape[0]
        bins = torch.linspace(0.0, 1.0, num_samples + 1).to(ray_bundle.origins.device)[None, ...]
        if self.train_stratified and self.training:
            if self.single_jitter:
                t_rand = torch.rand((num_rays, 1), dtype=bins.dtype, device=bins.device)
            else:
                t_rand = torch.rand((num_rays, num_samples + 1), dtype=bins.dtype, device=bins.device)
            bin_centers = (bins[..., 1:] + bins[..., :-1]) / 2.0
            bin_upper = torch.cat([bin_centers, bins[..., -1:]], -1)
            bin_lower = torch.cat([bins[..., :1], bin_centers], -1)
            bins = bin_lower + (bin_upper - bin_lower) * t_rand
        s_near, s_far = (self.spacing_fn(x) for x in (ray_bundle.nears, ray_bundle.fars))

        def spacing_to_euclidean_fn(x):
            return self.spacing_fn_inv(x * s_far + (1 - x) * s_near)

        euclidean_bins = spacing_to_euclidean_fn(bins)
        ray_samples = ray_bundle.get_ray_samples(bin_starts=euclidean_bins[..., :-1, None], bin_ends=euclidean_bins[..., 1:, None], spacing_starts=bins[..., :-1, None],
                                                 spacing_ends=bins[..., 1:, None], spacing_to_euclidean_fn=spacing_to_euclidean_fn)
        return ray_samples