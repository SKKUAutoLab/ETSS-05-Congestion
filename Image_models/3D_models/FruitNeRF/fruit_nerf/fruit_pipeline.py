from __future__ import annotations
import typing
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Dict, List, Literal, Optional, Type
import torch
import torch.distributed as dist
from PIL import Image
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from torch.nn import Parameter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp.grad_scaler import GradScaler
from nerfstudio.configs import base_config as cfg
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, VanillaDataManager
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import profiler
from nerfstudio.pipelines.base_pipeline import Pipeline

@dataclass
class FruitPipelineConfig(cfg.InstantiateConfig):
    _target: Type = field(default_factory=lambda: FruitPipeline)
    datamanager: DataManagerConfig = DataManagerConfig()
    model: ModelConfig = ModelConfig()

class FruitPipeline(Pipeline):
    def __init__(self, config: FruitPipelineConfig, device: str, test_mode: Literal["test", "val", "inference"] = "val", world_size: int = 1, local_rank: int = 0, grad_scaler: Optional[GradScaler] = None):
        super().__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank)
        self.datamanager.to(device)
        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        self._model = config.model.setup(scene_box=self.datamanager.train_dataset.scene_box, num_train_data=len(self.datamanager.train_dataset),
                                         metadata=self.datamanager.train_dataset.metadata, device=device, grad_scaler=grad_scaler, test_mode=test_mode, render_rgb_inference=True)
        self.model.to(device)
        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self._model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        if self.config.datamanager.camera_optimizer is not None:
            camera_opt_param_group = self.config.datamanager.camera_optimizer.param_group
            if camera_opt_param_group in self.datamanager.get_param_groups():
                metrics_dict["camera_opt_translation"] = (self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, :3].norm())
                metrics_dict["camera_opt_rotation"] = (self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, 3:].norm())
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        return model_outputs, loss_dict, metrics_dict

    def forward(self):
        raise NotImplementedError

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        self.eval()
        image_idx, camera_ray_bundle, batch = self.datamanager.next_eval_image(step)
        outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        assert "image_idx" not in metrics_dict
        metrics_dict["image_idx"] = image_idx
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = len(camera_ray_bundle)
        self.train()
        return metrics_dict, images_dict

    @profiler.time_function
    def get_average_eval_image_metrics(self, step: Optional[int] = None, output_path: Optional[Path] = None):
        self.eval()
        metrics_dict_list = []
        assert isinstance(self.datamanager, VanillaDataManager)
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn(), MofNCompleteColumn(), transient=True) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for camera_ray_bundle, batch in self.datamanager.fixed_indices_eval_dataloader:
                inner_start = time()
                height, width = camera_ray_bundle.shape
                num_rays = height * width
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
                if output_path is not None:
                    camera_indices = camera_ray_bundle.camera_indices
                    assert camera_indices is not None
                    for key, val in images_dict.items():
                        Image.fromarray((val * 255).byte().cpu().numpy()).save(output_path / "{0:06d}-{1}.jpg".format(int(camera_indices[0, 0, 0]), key))
                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            metrics_dict[key] = float(torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])))
        self.train()
        return metrics_dict

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        state = {(key[len("module."):] if key.startswith("module.") else key): value for key, value in loaded_state.items()}
        self.model.update_to_step(step)
        self.load_state_dict(state, strict=True)

    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes) -> List[TrainingCallback]:
        datamanager_callbacks = self.datamanager.get_training_callbacks(training_callback_attributes)
        model_callbacks = self.model.get_training_callbacks(training_callback_attributes)
        callbacks = datamanager_callbacks + model_callbacks
        return callbacks

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        datamanager_params = self.datamanager.get_param_groups()
        model_params = self.model.get_param_groups()
        return {**datamanager_params, **model_params}