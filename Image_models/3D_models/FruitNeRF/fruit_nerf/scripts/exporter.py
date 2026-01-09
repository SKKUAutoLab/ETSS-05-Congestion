from __future__ import annotations
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union
import open3d as o3d
import torch
import tyro
from typing_extensions import Annotated
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.scripts.exporter import ExportPointCloud
from fruit_nerf.export.exporter_utils import sample_volume

@dataclass
class Exporter:
    load_config: Path
    output_dir: Path

@dataclass
class ExportSemanticPointCloud(Exporter):
    use_bounding_box: bool = True
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    num_rays_per_batch: int = 32768
    num_points_per_side: int = 1000

    def main(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
        config, pipeline, _, _ = eval_setup(self.load_config, test_mode='export')
        assert isinstance(pipeline.datamanager, VanillaDataManager)
        pipeline.datamanager.config.eval_num_rays_per_batch = self.num_rays_per_batch
        pipeline.model.setup_inference(render_rgb=True, num_inference_samples=self.num_points_per_side)
        num_points = pipeline.datamanager.setup_inference(num_points=self.num_points_per_side, aabb=(self.bounding_box_min, self.bounding_box_max))
        with open(self.load_config.parent / 'dataparser_transforms.json', 'r') as fp:
            transform_json = json.load(fp)
        pcds = sample_volume(pipeline=pipeline, num_points=num_points, output_dir=self.output_dir, config=config, transform_json=transform_json)
        torch.cuda.empty_cache()
        os.makedirs(str(self.output_dir / config.load_dir.parts[-3]), exist_ok=True)
        CONSOLE.print("Saving Point Cloud...")
        for pcd_name in pcds.keys():
            pcd = pcds[pcd_name]['pcd']
            pcd_path = pcds[pcd_name]['path']
            o3d.io.write_point_cloud(pcd_path, pcd)
        CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")

Commands = tyro.conf.FlagConversionOff[Union[Annotated[ExportSemanticPointCloud, tyro.conf.subcommand(name="semantic-pointcloud")], Annotated[ExportPointCloud, tyro.conf.subcommand(name="pointcloud")]]]

def entrypoint():
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()

if __name__ == "__main__":
    entrypoint()

def get_parser_fn():
    return tyro.extras.get_parser(Commands)