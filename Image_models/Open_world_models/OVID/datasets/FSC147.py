# https://github.com/Verg-Avesta/CounTR/blob/main/util/FSC147.py
import json
import numpy as np
import random
from torchvision import transforms
import torch
import torchvision.transforms.functional as TF
import scipy.ndimage as ndimage
from PIL import Image
from torch.utils.data import Dataset
import os
import imgaug as ia
import imgaug.augmenters as iaa
import pickle
from imgaug.augmentables import Keypoint, KeypointsOnImage

TTensor = transforms.Compose([transforms.ToTensor()])
Augmentation = transforms.Compose([transforms.ColorJitter(brightness=0.3, contrast=0.15, saturation=0.2, hue=0.2), transforms.GaussianBlur(kernel_size=(7, 9))])

class FSC147(Dataset):
    def __init__(self, split: str, subset_scale: float=1.0, resize_val: bool=True, additional_prompt: bool=True):
        assert split in ['train', 'val', 'test' , 'val_coco', 'test_coco']
        self.data_dir = "data/FSC147"
        self.dataset_type = 'FSC147'
        self.resize_val = resize_val
        self.im_dir = os.path.join(self.data_dir,'images_384_VarV2')
        self.gt_dir = os.path.join(self.data_dir, 'gt_density_map_adaptive_384_VarV2')
        self.anno_file = os.path.join(self.data_dir , f'annotation_{self.dataset_type}_384.json')
        self.data_split_file = os.path.join(self.data_dir ,f'Train_Test_Val_{self.dataset_type}.json')
        self.class_file = os.path.join(self.data_dir ,f'ImageClasses_{self.dataset_type}.txt')
        self.split = split
        with open(self.data_split_file) as f:
            data_split = json.load(f)
        with open(self.anno_file) as f:
            self.annotations = json.load(f)
        self.idx_running_set = data_split[split]
        self.idx_running_set = self.idx_running_set[:int(subset_scale*len(self.idx_running_set))]
        self.class_dict = {}
        with open(self.class_file) as f:
            for line in f:
                key = line.split()[0]
                val = line.split()[1:]
                val = ' '.join(val)
                self.class_dict[key] = val
        self.all_classes = list(set(self.class_dict.values()))
        self.transform = None
        if self.split == 'train' or (self.split == 'val' and self.resize_val):
            use_aug = self.split == 'train'
            self.transform = transforms.Compose([ResizeTrainImage(self, aug=use_aug)])
        random.shuffle(self.idx_running_set)
        self.additional_prompt = None
        self.use_additional_prompt = additional_prompt
        if additional_prompt:
            additional_prompt_path = "datasets/CLIP_caption.pkl"
            with open(additional_prompt_path, 'rb') as f:
                self.additional_prompt = pickle.load(f)

    def __len__(self):
        return len(self.idx_running_set)

    def __getitem__(self, idx):
        im_id = self.idx_running_set[idx]
        anno = self.annotations[im_id]
        text = self.class_dict[im_id]
        if self.use_additional_prompt:
            additional_prompt = self.additional_prompt[im_id]
        bboxes = anno['box_examples_coordinates']
        if self.split == 'train' or (self.split == 'val' and self.resize_val):
            rects = list()
            for bbox in bboxes:
                x1 = bbox[0][0]
                y1 = bbox[0][1]
                x2 = bbox[2][0]
                y2 = bbox[2][1]
                rects.append([y1, x1, y2, x2])
            dots = np.array(anno['points'])
            image = Image.open('{}/{}'.format(self.im_dir, im_id))
            image.load()
            density_path = self.gt_dir + '/' + im_id.split(".jpg")[0] + ".npy"
            density = np.load(density_path).astype('float32')   
            m_flag = 0
            sample = {'image':image,'lines_boxes':rects,'gt_density':density, 'dots':dots, 'id':im_id, 'm_flag': m_flag}
            sample = self.transform(sample)
            if self.use_additional_prompt:
                return sample['image'].float(), sample['gt_density'],sample['dots'], sample['boxes'], sample['m_flag'], text, additional_prompt
            return sample['image'].float(), sample['gt_density'],sample['dots'], sample['boxes'], sample['m_flag'], text
        elif self.split == "test" or self.split == "test_coco" or self.split == "val_coco" or (self.split == "val" and not self.resize_val):
            dots = np.array(anno['points'])
            image = Image.open('{}/{}'.format(self.im_dir, im_id))
            text = self.class_dict[im_id]
            image.load()
            W, H = image.size
            new_H = 16 * int(H / 16)
            new_W = 16 * int(W / 16)
            scale_factor = float(new_W) / W
            image = transforms.Resize((new_H, new_W))(image)
            Normalize = transforms.Compose([transforms.ToTensor()])
            image = Normalize(image)
            rects = list()
            for bbox in bboxes:
                x1 = int(bbox[0][0]*scale_factor)
                y1 = bbox[0][1]
                x2 = int(bbox[2][0]*scale_factor)
                y2 = bbox[2][1]
                rects.append([y1, x1, y2, x2])
            boxes = list()
            cnt = 0
            for box in rects:
                cnt += 1
                if cnt > 3:
                    break
                box2 = [int(k) for k in box]
                y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
                bbox = image[:,y1:y2 + 1,x1:x2 + 1]
                bbox = transforms.Resize((64, 64))(bbox)
                boxes.append(bbox.numpy())
            boxes = np.array(boxes)
            boxes = torch.Tensor(boxes)
            gt_map = np.zeros((image.shape[1], image.shape[2]),dtype='float32')
            for i in range(dots.shape[0]):
                gt_map[min(new_H-1,int(dots[i][1]))][min(new_W-1,int(dots[i][0]*scale_factor))]=1
            dots*=(scale_factor,1)
            gt_map = ndimage.gaussian_filter(gt_map, sigma=(1, 1), order=0)
            gt_map = torch.from_numpy(gt_map)
            gt_map = gt_map * 60
            sample = {'image': image,'dots': {'point': torch.Tensor(dots), 'labels': torch.ones(dots.shape[0]).long()}, 'boxes': boxes, 'pos': rects, 'gt_map': gt_map}
            return sample['image'].float(), sample['gt_map'],sample['dots'],  sample['boxes'], sample['pos'], text

class ResizeTrainImage(object):
    def __init__(self, dataset:FSC147=None, aug = True):
        self.dataset = dataset
        self.use_out_mosaic = False
        self.use_augmentation = aug

    def __call__(self, sample):
        image, lines_boxes, density, dots, im_id, m_flag = sample['image'], sample['lines_boxes'],sample['gt_density'], sample['dots'], sample['id'], sample['m_flag']
        W, H = image.size
        new_H = 16 * int(H / 16)
        new_W = 16 * int(W / 16)
        scale_factor = float(new_W)/ W
        resized_image = transforms.Resize((new_H, new_W))(image)
        dots = dots * (scale_factor, float(H) / H)
        aug_p = random.random()
        aug_flag = 0
        mosaic_flag = 0
        if self.use_augmentation and aug_p < 0.5:
            aug_flag = 1
            if aug_p < 0.3:
                aug_flag = 0
                mosaic_flag = 1
        resized_image = TTensor(resized_image)
        if aug_flag == 1:
            noise = np.random.normal(0, 0.1, resized_image.size())
            noise = torch.from_numpy(noise)
            re_image = resized_image + noise
            re_image = torch.clamp(re_image, 0, 1)
        if aug_flag == 1:
            re_image = Augmentation(re_image)
        if aug_flag == 1:
            re1_image = re_image.transpose(0,1).transpose(1,2).numpy()
            keypoints = []
            for i in range(dots.shape[0]):
                keypoints.append(Keypoint(x=min(new_W-1,int(dots[i][0])), y=min(new_H-1,int(dots[i][1]))))
            kps = KeypointsOnImage(keypoints, re1_image.shape)
            seq = iaa.Sequential([iaa.Affine(rotate=(-15,15), scale=(0.8, 1.2), shear=(-10,10), translate_percent={"x": (-0.2,0.2), "y": (-0.2,0.2)}, mode=ia.ALL)])
            re1_image, kps_aug = seq(image=re1_image, keypoints=kps)
            dots = np.array([(kp.x, kp.y) for kp in kps_aug if not kp.is_out_of_image(re1_image)])
            if dots.ndim != 2:
                dots=np.empty((0, 2))
            else:
                idx = (dots[:, 0] <= new_W)& (dots[:, 1] <= new_H)
                record_den = dots[idx]
                dots = record_den
            re_image = TTensor(re1_image)
        if aug_flag == 1:
            flip_p = random.random()
            if flip_p > 0.5:
                re_image = TF.hflip(re_image)
                dots[:, 0] = new_W-dots[:, 0]
        if aug_flag == 1:
            flip_p = random.random()
            if flip_p > 0.5:
                re_image = TF.vflip(re_image)
                dots[:, 1] = new_H - dots[:, 1]
        if mosaic_flag == 0:
            if aug_flag == 0:
                re_image = resized_image
            start = random.randint(0, new_W-1-383)
            reresized_image = TF.crop(re_image, 0, start, 384, 384)
            start_w = start
            end_w = start + 384
            idx = (dots[:, 0] >= start_w) & (dots[:, 0] <= end_w) &(dots[:, 1] >= 0) & (dots[:, 1] <= 384)
            record_den = dots[idx]
            record_den[:, 0] -= start_w
            dots = record_den
        else:
            image_array = []
            dots_array=[]
            blending_l = random.randint(10, 20)
            resize_l = 192 + 2 * blending_l
            if dots.shape[0] >= 70 or not self.use_out_mosaic:
                for i in range(4):
                    length =  random.randint(150, 384)
                    start_W = random.randint(0, new_W-length)
                    start_H = random.randint(0, new_H-length)
                    reresized_image1 = TF.crop(resized_image, start_H, start_W, length, length)
                    reresized_image1 = transforms.Resize((resize_l, resize_l))(reresized_image1)
                    reresized_image1 = Augmentation(reresized_image1)
                    end_w = start_W + length
                    end_h = start_H + length
                    dots_scale=float(resize_l)/length
                    idx = (dots[:, 0] >= start_W) & (dots[:, 0] <= end_w) & (dots[:, 1] >= start_H) & (dots[:, 1] <= end_h)
                    record_den = dots[idx]
                    record_den[:, 0] -= start_W
                    record_den[:, 1] -= start_H
                    record_den *= (dots_scale, dots_scale)
                    image_array.append(reresized_image1)
                    dots_array.append(record_den)
            elif self.use_out_mosaic:
                m_flag = 1
                prob = random.random()
                if prob > 0.25:
                    gt_pos = random.randint(0,3)
                else:
                    gt_pos = random.randint(0,4)
                for i in range(4):
                    if i == gt_pos:
                        r_image = resized_image
                        Tdots = dots
                        new_TH = new_H
                        new_TW = new_W
                    else:
                        Tim_id = self.dataset.idx_running_set[random.randint(0, len(self.dataset.idx_running_set)-1)]
                        Tdots = np.array(self.dataset.annotations[Tim_id]['points'])
                        Timage = Image.open('{}/{}'.format(self.dataset.im_dir, Tim_id))
                        Timage.load()
                        new_TH = 16*int(Timage.size[1] / 16)
                        new_TW = 16*int(Timage.size[0] / 16)
                        Tscale_factor = float(new_TW) / Timage.size[0]
                        r_image = TTensor(transforms.Resize((new_TH, new_TW))(Timage))
                        Tdots = Tdots * (Tscale_factor,float(new_TH) / Timage.size[1])
                    length =  random.randint(250, 384)
                    start_W = random.randint(0, new_TW-length)
                    start_H = random.randint(0, new_TH-length)
                    r_image1 = TF.crop(r_image, start_H, start_W, length, length)
                    r_image1 = transforms.Resize((resize_l, resize_l))(r_image1)
                    dots_scale=float(resize_l)/length
                    end_W = start_W + length
                    end_H = start_H + length
                    idx = (Tdots[:, 0] >= start_W) & (Tdots[:, 0] <= end_W) & (Tdots[:, 1] >= start_H) & (Tdots[:, 1] <= end_H)
                    record_den = Tdots[idx]
                    record_den[:, 0] -= start_W
                    record_den[:, 1] -= start_H
                    record_den *= (dots_scale, dots_scale)
                    image_array.append(r_image1)
                    dots_array.append(record_den)
            reresized_image5 = torch.cat((image_array[0][:, blending_l:resize_l - blending_l],image_array[1][:,blending_l:resize_l - blending_l]),1)
            start_W = blending_l
            end_W = resize_l - blending_l
            start_H = blending_l
            end_H = resize_l - blending_l
            for i in range(len(dots_array)):
                idx =(dots_array[i][:, 0] >= start_W) & (dots_array[i][:, 0] <= end_W)&(dots_array[i][:, 1] >= start_H) & (dots_array[i][:, 1] <= end_H)
                record_den = dots_array[i][idx]
                record_den[:, 0] -= start_W
                record_den[:, 1] -= start_H
                if i % 2==1:
                    record_den[:, 1] += 192
                if i > 1:
                    record_den[:, 0] +=192
                dots_array[i] = record_den
            for i in range(blending_l):
                reresized_image5[:,192+i] = image_array[0][:,resize_l-1-blending_l+i] * (blending_l-i)/(2*blending_l) + reresized_image5[:,192+i] * (i+blending_l)/(2*blending_l)
                reresized_image5[:,191-i] = image_array[1][:,blending_l-i] * (blending_l-i)/(2*blending_l) + reresized_image5[:,191-i] * (i+blending_l)/(2*blending_l)
            reresized_image5 = torch.clamp(reresized_image5, 0, 1)
            reresized_image6 = torch.cat((image_array[2][:,blending_l:resize_l-blending_l],image_array[3][:,blending_l:resize_l-blending_l]),1)
            for i in range(blending_l):
                reresized_image6[:,192+i] = image_array[2][:,resize_l-1-blending_l+i] * (blending_l-i)/(2*blending_l) + reresized_image6[:,192+i] * (i+blending_l)/(2*blending_l)
                reresized_image6[:,191-i] = image_array[3][:,blending_l-i] * (blending_l-i)/(2*blending_l) + reresized_image6[:,191-i] * (i+blending_l)/(2*blending_l)
            reresized_image6 = torch.clamp(reresized_image6, 0, 1)
            reresized_image = torch.cat((reresized_image5[:,:,blending_l:resize_l-blending_l],reresized_image6[:,:,blending_l:resize_l-blending_l]),2)
            for i in range(blending_l):
                reresized_image[:,:,192+i] = reresized_image5[:,:,resize_l-1-blending_l+i] * (blending_l-i)/(2*blending_l) + reresized_image[:,:,192+i] * (i+blending_l)/(2*blending_l)
                reresized_image[:,:,191-i] = reresized_image6[:,:,blending_l-i] * (blending_l-i)/(2*blending_l) + reresized_image[:,:,191-i] * (i+blending_l)/(2*blending_l)
            reresized_image = torch.clamp(reresized_image, 0, 1)
            dots = np.vstack(tuple(dots_array))
        resized_density = np.zeros(shape=(reresized_image.shape[2],reresized_image.shape[1]),dtype='float32')
        for i in range(dots.shape[0]):
            resized_density[min(reresized_image.shape[2] - 1, int(dots[i][1]))][min(reresized_image.shape[1] - 1, int(dots[i][0]))] = 1
        point_density=resized_density
        reresized_density = ndimage.gaussian_filter(resized_density, sigma=(1, 1), order=0)
        reresized_density = reresized_density * 60
        reresized_density = torch.from_numpy(reresized_density)
        boxes = list()
        cnt = 0
        for box in lines_boxes:
            cnt += 1
            if cnt > 3:
                break
            box2 = [int(k) for k in box]
            y1, x1, y2, x2 = box2[0], int(box2[1] * scale_factor), box2[2], int(box2[3] * scale_factor)
            bbox = resized_image[:, y1:y2 + 1, x1:x2 + 1]
            bbox = transforms.Resize((64, 64))(bbox)
            boxes.append(bbox.numpy())
        boxes = np.array(boxes)
        boxes = torch.Tensor(boxes)
        sample = {'image': reresized_image, 'gt_density': reresized_density, 'boxes': boxes,'dots': {'point': torch.Tensor(dots), 'labels': torch.ones(dots.shape[0]).long()},
                  'm_flag': m_flag,'ground_point':point_density}
        return sample