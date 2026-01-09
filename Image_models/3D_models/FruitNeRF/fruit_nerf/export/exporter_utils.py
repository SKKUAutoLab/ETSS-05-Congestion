from __future__ import annotations
import numpy as np
import open3d as o3d
import torch
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.rich_utils import CONSOLE
import pathlib

def sample_volume(pipeline: Pipeline, num_points: int, output_dir: pathlib.Path = None, config=None, transform_json: dict = None) -> dict:
    progress = Progress(TextColumn(":cloud: Computing Point Cloud :cloud:"), BarColumn(), TaskProgressColumn(show_speed=True),
                        TimeRemainingColumn(elapsed_when_finished=True, compact=True), console=CONSOLE)
    points_sem = []
    points_only_sem = []
    points_den = []
    points_sem_colormap = []
    color_semantics = []
    color_only_semantics = []
    color_semantics_colormap = []
    densities = []
    rgb_flag = True
    with progress as progress_bar:
        task = progress_bar.add_task("Generating Point Cloud", total=num_points)
        while not progress_bar.finished:
            with torch.no_grad():
                ray_bundle, _ = pipeline.datamanager.next_sample_volume(0)
                outputs = pipeline.model(ray_bundle)
            sampled_point_position = outputs['point_location']
            points_3d = sampled_point_position.reshape((-1, 3))
            semantic = outputs['semantics'].reshape((-1, 1)).repeat((1, 3))
            semantics_colormap = outputs['semantics_colormap'].reshape((-1, 1)).repeat((1, 3))
            density = outputs['density'].reshape((-1, 1)).repeat((1, 3))
            rgb = outputs['rgb'].reshape((-1, 3))
            mask_sem = semantic >= 3
            mask_den = density >= 70
            mask_sem_colormap = semantics_colormap >= 0.999
            mask_only_sem = semantics_colormap >= 0.99
            points_3d_semantic_colormap = points_3d[
                mask_sem_colormap.sum(dim=1).to(bool) & mask_den.sum(dim=1).to(bool)]
            if rgb_flag:
                color_semantic_colormap = rgb[mask_sem_colormap.sum(dim=1).to(bool) & mask_den.sum(dim=1).to(bool)]
            else:
                color_semantic_colormap = semantics_colormap[mask_sem_colormap.sum(dim=1).to(bool) & mask_den.sum(dim=1).to(bool)]
            color_semantic_colormap = torch.hstack([color_semantic_colormap, torch.sigmoid(semantic[mask_sem_colormap.sum(dim=1).to(bool) & mask_den.sum(dim=1).to(bool)][:, 0]).unsqueeze(-1)])
            points_sem_colormap.append(points_3d_semantic_colormap.cpu())
            color_semantics_colormap.append(color_semantic_colormap.cpu())
            points_3d_semantic = points_3d[mask_sem.sum(dim=1).to(bool) & mask_den.sum(dim=1).to(bool)]
            if rgb_flag:
                color_semantic = rgb[mask_sem.sum(dim=1).to(bool) & mask_den.sum(dim=1).to(bool)]
            else:
                color_semantic = semantic[mask_sem.sum(dim=1).to(bool) & mask_den.sum(dim=1).to(bool)]
            color_semantic = torch.hstack([color_semantic, torch.sigmoid(
                semantic[mask_sem.sum(dim=1).to(bool) & mask_den.sum(dim=1).to(bool)][:, 0]).unsqueeze(-1)])
            points_sem.append(points_3d_semantic.cpu())
            color_semantics.append(color_semantic.cpu())
            points_3d_density = points_3d[mask_den.sum(dim=1).to(bool)]
            if rgb_flag:
                density_color = rgb[mask_den.sum(dim=1).to(bool)]
            else:
                density_color = density[mask_den.sum(dim=1).to(bool)]
            density_color = torch.hstack([density_color, torch.sigmoid(density[mask_den.sum(dim=1).to(bool)][:, 0]).unsqueeze(-1)])
            points_den.append(points_3d_density.cpu())
            densities.append(density_color.cpu())
            if False:
                points_3d_only_semantic_colormap = points_3d[mask_only_sem.sum(dim=1).to(bool)]
                if rgb_flag:
                    sem_color_only = rgb[mask_only_sem.sum(dim=1).to(bool)]
                else:
                    sem_color_only = semantic[mask_only_sem.sum(dim=1).to(bool)]
                points_only_sem.append(points_3d_only_semantic_colormap.cpu())
                color_only_semantics.append(sem_color_only.cpu())
            torch.cuda.empty_cache()
            progress.advance(task, sampled_point_position.shape[0])
    pcd_list = {}
    points_sem_colormap = torch.cat(points_sem_colormap, dim=0)
    semantic_colormap_rgbs = torch.cat(color_semantics_colormap, dim=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_sem_colormap.detach().double().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(semantic_colormap_rgbs.detach().double().cpu().numpy()[:, :3])
    if True:
        T = np.eye(4)
        T[:3, :4] = np.asarray(transform_json['transform'])[:3, :4]
        T[:3, :3] = T[:3, :3]
        T[:3, 3] *= -1
        pcd = pcd.scale(1 / transform_json['scale'], center=np.asarray((0, 0, 0)))
        pcd = pcd.scale(2, center=np.asarray((0, 0, 0)))
    pcd_list.update({'semantic_colormap': {'pcd': pcd, 'path': str(output_dir / config.load_dir.parts[-3] / 'semantic_colormap.ply')}})
    points_sem = torch.cat(points_sem, dim=0)
    semantic_rgbs = torch.cat(color_semantics, dim=0)
    if semantic_rgbs.shape[0] != 0:
        semantic_rgbs /= semantic_rgbs.max()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_sem.double().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(semantic_rgbs.double().cpu().numpy()[:, :3])
    if True:
        T = np.eye(4)
        T[:3, :4] = np.asarray(transform_json['transform'])[:3, :4]
        T[:3, :3] = T[:3, :3]
        T[:3, 3] *= -1
        pcd = pcd.scale(1 / transform_json['scale'], center=np.asarray((0, 0, 0)))
        pcd = pcd.scale(2, center=np.asarray((0, 0, 0)))
    pcd_list.update({'semantic': {'pcd': pcd, 'path': str(output_dir / config.load_dir.parts[-3] / 'semantic.ply')}})
    points_den = torch.cat(points_den, dim=0)
    density_rgb = torch.cat(densities, dim=0)
    if density_rgb.shape[0] != 0:
        density_rgb /= density_rgb.max()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_den.double().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(density_rgb.double().cpu().numpy()[:, :3])
    if True:
        T = np.eye(4)
        T[:3, :4] = np.asarray(transform_json['transform'])[:3, :4]
        T[:3, :3] = T[:3, :3]
        T[:3, 3] *= -1
        pcd = pcd.scale(1 / transform_json['scale'], center=np.asarray((0, 0, 0)))
        pcd = pcd.scale(2, center=np.asarray((0, 0, 0)))
    pcd_list.update({'density': {'pcd': pcd, 'path': str(output_dir / config.load_dir.parts[-3] / 'density.ply')}})
    return pcd_list