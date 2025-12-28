from __future__ import annotations
import random
from torchvision.datasets import VisionDataset
from PIL import Image
import os
from typing import Any, Callable, Tuple
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

def transform():
    return A.Compose([A.Resize(720, 1280), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])

def video_transform():
    # return A.Compose([A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]) # use for > 128GB ram
    return A.Compose([A.Resize(720, 1280), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]) # use for <= 128GB ram

class VideoDataset(VisionDataset):
    def __init__(self, root: str, transforms: Callable[..., Any] | None = None, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.video_paths = os.listdir(root)
        self.video_paths = [os.path.join(root, video_path, 'img1') for video_path in self.video_paths]
        self.annotations_paths = os.listdir(root)
        self.annotations_paths = [os.path.join(root, annotation, 'det/det.txt') for annotation in self.annotations_paths]
        self.annotation = {}
        self.videos = []
        width, height = 1920, 1080
        for annotation_path in self.annotations_paths:
            with open(annotation_path) as f:
                video_name = annotation_path.split('/')[-3]
                self.annotation[video_name] = []
                lines = f.readlines()
                file_idx = 0
                cnt = 0
                for line in lines:
                    line = line.split(',')
                    data = [float(x) for x in line[0:] if x != ""]
                    data = np.array(data)
                    if file_idx != int(line[0]):
                        if file_idx != 0:
                            self.annotation[video_name].append({"file_name": file_name, "pts":points, "height": height, "width": width, "cnt": cnt})
                        file_idx = int(line[0])
                        file_name = str(file_idx).zfill(6) + '.jpg'
                        cnt = 1
                        if len(data) > 0:
                            bbox = data[2:6]
                            bbox[0] = bbox[0] / width
                            bbox[1] = bbox[1] / height
                            bbox[2] = bbox[2] / width
                            bbox[3] = bbox[3] / height
                            bboxes = np.array([bbox])
                            points = np.array(bboxes[:, 0:2] + bboxes[:, 2:4] / 2)
                        else:
                            bboxes = -1 * np.ones((1, 4))
                    else:
                        cnt += 1
                        if len(data) > 0:
                            bbox = data[2:6]
                            bbox[0] = bbox[0] / width
                            bbox[1] = bbox[1] / height
                            bbox[2] = bbox[2] / width
                            bbox[3] = bbox[3] / height
                            bboxes = np.concatenate((bboxes, np.array([bbox])), axis=0)
                            pts = bbox[0:2] + bbox[2:4] / 2
                            points = np.concatenate((points, np.array([pts])), axis=0)
                        else:
                            points = -1 * np.ones((1, 2))
                            bboxes = -1 * np.ones((1, 4))
        for video_path in self.video_paths:
            video_name = video_path.split('/')[-2]
            self.videos.append({"video_name": video_name, "height": self.annotation[video_name][0]["height"], "width": self.annotation[video_name][0]["width"], "cnt": [],
                                "img_names": [], "video_path": video_path, "pts": []})
            for i in range(0, len(self.annotation[video_name])):
                self.videos[-1]["img_names"].append(self.annotation[video_name][i]["file_name"])
                self.videos[-1]["pts"].append(self.annotation[video_name][i]["pts"])
                self.videos[-1]["cnt"].append(self.annotation[video_name][i]["cnt"])

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        video = self.videos[index]
        video_name = video["video_name"]
        img_names = video["img_names"]
        height = video["height"]
        width = video["width"]
        cnt = video["cnt"]
        points = video["pts"]
        video_path = video["video_path"]
        imgs = []
        for img_name in img_names:
            img_path = os.path.join(self.root, video_name, 'img1', img_name)
            img = self.transforms(image=np.array(Image.open(img_path).convert("RGB")))["image"]
            imgs.append(img)
        imgs = torch.stack(imgs, dim=0)
        labels = {"h": height, "w": width, "cnt": cnt, "video_name": video_name, "img_names": img_names, "pts": points, "video_path": video_path}
        return imgs, labels

class PairDataset(VisionDataset):
    def __init__(self, root: str, max_len: int, transforms: Callable[..., Any] | None = None, transform: Callable[..., Any] | None = None,
                 target_transform: Callable[..., Any] | None = None, train=True, step=20, interval=1, force_last=False) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.video_paths = os.listdir(root)
        self.video_paths = [os.path.join(root, video_path,'img1') for video_path in self.video_paths]
        self.annotations_paths = os.listdir(root)
        self.annotations_paths = [os.path.join(root, annotation, 'gt/gt.txt') for annotation in self.annotations_paths]
        self.annotation = {}
        self.pairs = []
        self.max_len = max_len
        width, height = 1920, 1080
        for annotation_path in self.annotations_paths:
            with open(annotation_path) as f:
                video_name = annotation_path.split('/')[-3]
                self.annotation[video_name] = []
                lines = f.readlines()
                file_idx = 0
                cnt = 0
                for line in lines:
                    line = line.split(',')
                    data = [float(x) for x in line[0:] if x != ""]
                    data = np.array(data)
                    if file_idx != int(line[0]):
                        if file_idx != 0:
                            self.annotation[video_name].append({"file_name": file_name, "height": height, "width": width, "ids": ids, "pts": points, "bboxes": bboxes, "cnt": cnt})
                        file_idx = int(line[0])
                        file_name = str(file_idx).zfill(6) + '.jpg'
                        cnt = 1
                        if len(data) > 0:
                            bbox = data[2:6]
                            bbox[0] = bbox[0] / width
                            bbox[1] = bbox[1] / height
                            bbox[2] = bbox[2] / width
                            bbox[3] = bbox[3] / height
                            bboxes = np.array([bbox])
                            points = np.array(bboxes[:,0:2] + bboxes[:,2:4] / 2)
                            ids = np.array([[data[1].astype(int)]])
                        else:
                            ids = -1 * np.ones((1, 1))
                            bboxes = -1 * np.ones((1, 4))
                    else:
                        cnt += 1
                        if len(data) > 0:
                            bbox = data[2:6]
                            bbox[0] = bbox[0] / width
                            bbox[1] = bbox[1] / height
                            bbox[2] = bbox[2] / width
                            bbox[3] = bbox[3] / height
                            bboxes = np.concatenate((bboxes, np.array([bbox])), axis=0)
                            pts = bbox[0:2] + bbox[2:4] / 2
                            points = np.concatenate((points, np.array([pts])), axis=0)
                            ids = np.concatenate((ids, np.array([[data[1].astype(int)]])), axis=0)
                        else:
                            ids = -1 * np.ones((1, 1))
                            points = -1 * np.ones((1, 2))
                            bboxes = -1 * np.ones((1, 4))
        for video_path in self.video_paths:
            video_name = video_path.split('/')[-2]
            last_step = 0
            for i in range(1, len(self.annotation[video_name]) - step, interval):
                self.pairs.append({"0": self.annotation[video_name][i], "1": self.annotation[video_name][i + step], "video_name": video_name})
                last_step = i + step
            if force_last and last_step < len(self.annotation[video_name]) - 1:
                self.pairs.append({"0": self.annotation[video_name][last_step], "1": self.annotation[video_name][-1], "video_name": video_name})
        self.train = train

    def __len__(self) -> int:
        return len(self.pairs)

    def add_noise(self, pts):
        noise = np.random.normal(scale=0.001, size=pts.shape)
        pts = pts + noise
        pts[pts > 1] = 1
        pts[pts < 0] = 0
        return pts

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        pair = self.pairs[index]
        img0_path = os.path.join(self.root, pair["video_name"],'img1', pair["0"]["file_name"])
        img1_path = os.path.join(self.root, pair["video_name"],'img1', pair["1"]["file_name"])
        video_name = pair["video_name"]
        img_name1 = pair["0"]["file_name"]
        img_name2 = pair["1"]["file_name"]
        cnt_0 = pair["0"]["cnt"]
        cnt_1 = pair["1"]["cnt"]
        pts_0 = pair["0"]["pts"]
        pts_1 = pair["1"]["pts"]
        if self.train:
            pts_0 = self.add_noise(pts_0)
            pts_1 = self.add_noise(pts_1)
        id_0 = pair["0"]["ids"]
        id_1 = pair["1"]["ids"]
        if (pair["0"]["height"] != pair["1"]["height"] or pair["0"]["width"] != pair["1"]["width"] or cnt_0 == 0 or cnt_1 == 0):
            print("error")
            print(pair["0"]["height"], pair["1"]["height"], pair["0"]["width"], pair["1"]["width"], cnt_0, cnt_1, video_name, img_name2)
            return self.__getitem__((index + 1) % len(self))
        img0 = self.transforms(image=np.array(Image.open(img0_path).convert("RGB")))["image"]
        img1 = self.transforms(image=np.array(Image.open(img1_path).convert("RGB")))["image"]
        fused_pts_list0 = []
        fused_pts_list1 = []
        id_list0 = []
        id_list1 = []
        fused_num = 0
        id1_pt1_dict = {id[0]: pt for id, pt in zip(id_1, pts_1)}
        independ0_list = []
        independ1_list = []
        for pt, id in zip(pts_0, id_0):
            if id in id_1:
                fused_pts_list0.append(pt)
                id_list0.append(id)
                fused_num += 1
                pt1 = id1_pt1_dict[id[0]]
                fused_pts_list1.append(pt1)
                id_list1.append(id)
            else:
                independ0_list.append(pt)
        for pt, id in zip(pts_1, id_1):
            if id not in id_0:
                independ1_list.append(pt)
        if len(independ0_list) == 0:
            independ0_list.append((0.25, 0.25))
        if len(independ1_list) == 0:
            independ1_list.append((0.75, 0.75))
        if self.train and (fused_num == 0 or len(independ0_list) == 0 or len(independ1_list) == 0):
            print("error")
            print(pair["0"]["height"], pair["1"]["height"], pair["0"]["width"], pair["1"]["width"], cnt_0, cnt_1, video_name, img_name2)
            return self.__getitem__((index + 1) % len(self))
        pt_0 = -1 * np.ones((self.max_len, 2))
        pt_1 = -1 * np.ones((self.max_len, 2))
        pt_0[:cnt_0] = pts_0
        pt_1[:cnt_1] = pts_1
        independ_pts0 = -1 * np.ones((self.max_len, 2))
        independ_pts0[:len(independ0_list)] = np.array(independ0_list)
        independ_pts1 = -1 * np.ones((self.max_len, 2))
        independ_pts1[:len(independ1_list)] = np.array(independ1_list)
        x = torch.cat([img0, img1], dim=0)
        fused_pts0 = -1 * np.ones((self.max_len, 2))
        fused_pts1 = -1 * np.ones((self.max_len, 2))
        if fused_num > 0:
            fused_pts0[:fused_num] = np.array(fused_pts_list0)
            fused_pts1[:fused_num] = np.array(fused_pts_list1)
        fused_pts1 = torch.from_numpy(fused_pts1).float()
        fused_pts0 = torch.from_numpy(fused_pts0).float()
        max_num = 120
        if self.train and fused_num >= max_num:
            mask = random.sample(range(fused_num), max_num)
            fused_pts0[:max_num] = fused_pts0[mask]
            fused_pts1[:max_num] = fused_pts1[mask]
            fused_num = max_num
        ref_pts = torch.stack([fused_pts0, fused_pts1], dim=0)
        labels = {"h": pair["0"]["height"], "w": pair["0"]["width"], "gt_fuse_pts0": fused_pts0, "gt_fuse_pts1": fused_pts1, "gt_default_num": pair["0"]["cnt"],
                  "gt_duplicate_num": pair["1"]["cnt"], "gt_fuse_num": fused_num, "video_name": video_name, "img_name1": img_name1, "img_name2": img_name2,
                  "cnt_0": cnt_0, "cnt_1": cnt_1, "pt_0": pt_0, "pt_1": pt_1}
        inputs = {"h": pair["0"]["height"], "w": pair["0"]["width"], "image_pair": x, "ref_pts": ref_pts, "ref_num": fused_num, "independ_pts0": torch.from_numpy(independ_pts0).float(),
                  "independ_pts1": torch.from_numpy(independ_pts1).float(), "independ_num0": len(independ0_list), "independ_num1": len(independ1_list), "cnt_0": cnt_0,
                  "cnt_1": cnt_1, "pt_0": pt_0, "pt_1": pt_1}
        return inputs, labels

def build_dataset(root, max_len, train=False, step=20, interval=1, force_last=False):
    transforms = transform()
    dataset = PairDataset(root, max_len, transforms=transforms, train=train, step=step, interval=interval, force_last=force_last)
    return dataset

def build_video_dataset(root):
    dataset = VideoDataset(root, transforms=video_transform())
    return dataset
