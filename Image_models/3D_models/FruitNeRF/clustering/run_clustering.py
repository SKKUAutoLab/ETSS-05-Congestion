from typing import Union
import open3d as o3d
import numpy as np
from pathlib import Path
from clustering_base import load_obj_file, FruitClustering
import sys
import os
import json
sys.path.append(os.getcwd())
from clustering.config_synthetic import (Apple_GT_1024x1024_300, Apple_SAM_1024x1024_300, Pear_GT_1024x1024_300, Pear_SAM_1024x1024_300, Plum_GT_1024x1024_300, Plum_SAM_1024x1024_300,
                                         Lemon_GT_1024x1024_300, Lemon_SAM_1024x1024_300, Peach_GT_1024x1024_300, Peach_SAM_1024x1024_300, Mango_GT_1024x1024_300, Mango_SAM_1024x1024_300)
from clustering.config_real import (Baum_01_unet, Baum_01_unet_Big, Baum_01_SAM, Baum_01_SAM_Big, Baum_02_unet, Baum_02_unet_Big, Baum_02_SAM, Baum_02_SAM_Big,
                                    Baum_03_unet, Baum_03_unet_Big, Baum_03_SAM, Baum_03_SAM_Big)
from clustering.config_real import Fuji_unet, Fuji_unet_big, Fuji_sam, Fuji_sam_big

class Clustering(FruitClustering):
    def __init__(self, template_path: Union[str, Path] = 'clustering/apple_template.ply', voxel_size_down_sample: float = 0.00005, remove_outliers_nb_points: int = 800,
                 remove_outliers_radius: float = 0.02, min_samples: int = 60, apple_template_size: float = 0.8, cluster_merge_distance: float = 0.04, gt_cluster=None, gt_count: int = None):
        super().__init__(voxel_size_down_sample=voxel_size_down_sample, remove_outliers_nb_points=remove_outliers_nb_points, remove_outliers_radius=remove_outliers_radius,
                         cluster_merge_distance=cluster_merge_distance)
        self.template_path = template_path
        self.min_samples = min_samples
        self.gt_cluster = gt_cluster
        if self.gt_cluster:
            if "obj" in self.gt_cluster:
                self.gt_mesh, self.gt_cluster_center, self.gt_cluster_pcd = load_obj_file(gt_cluster)
                self.gt_position = o3d.geometry.PointCloud()
                self.gt_position.points = o3d.utility.Vector3dVector(np.vstack(self.gt_cluster_center))
            else:
                self.gt_position = o3d.io.read_line_set(self.gt_cluster)
        self.gt_count = gt_count


if __name__ == '__main__':
    Baums = [# Fuji_unet, Fuji_unet_big, Fuji_sam, Fuji_sam_big, Baum_01_unet, Baum_01_unet_Big, Baum_01_SAM, Baum_01_SAM_Big, Baum_02_unet, Baum_02_unet_Big, Baum_02_SAM,
             # Baum_02_SAM_Big, Baum_03_unet, Baum_03_unet_Big, Baum_03_SAM, Baum_03_SAM_Big, Pear_GT_1024x1024_300, Pear_SAM_1024x1024_300, Plum_GT_1024x1024_300,
             # Plum_SAM_1024x1024_300, Lemon_GT_1024x1024_300, Lemon_SAM_1024x1024_300, Peach_GT_1024x1024_300, Peach_SAM_1024x1024_300,
             Apple_GT_1024x1024_300, # Apple_SAM_1024x1024_300, Mango_GT_1024x1024_300, Mango_SAM_1024x1024_300
    ]
    results = {}
    for Baum in Baums:
        clustering = Clustering(remove_outliers_nb_points=Baum['remove_outliers_nb_points'], remove_outliers_radius=Baum['remove_outliers_radius'],
                                voxel_size_down_sample=Baum['down_sample'], template_path=Baum['template_path'], min_samples=Baum['min_samples'],
                                apple_template_size=Baum['apple_template_size'], gt_cluster=Baum['gt_cluster'], cluster_merge_distance=Baum['cluster_merge_distance'],
                                gt_count=Baum['gt_count'])
        count = clustering.count(pcd=Baum["path"], eps=Baum['eps'])
        if Baum['gt_cluster']:
            results.update({Baum['path']: {'count': count, 'gt': clustering.gt_count}})
        else:
            results.update({Baum['path']: {'count': count, 'gt': clustering.gt_count}})
        print(results)
        print("\n --------------------------------- \n")
    print(results)
    with open('clustering/results_synthetic.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4, default=str)