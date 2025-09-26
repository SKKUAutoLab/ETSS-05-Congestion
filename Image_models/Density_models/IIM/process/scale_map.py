from torch.autograd import Variable
from  tqdm import  tqdm
import cv2 as cv
from models.CC import CrowdCounter
import os
import torch
import torchvision.transforms as standard_transforms
from PIL import Image
import numpy as np

mean_std = ([0.42968395352363586, 0.4371049106121063, 0.4219788610935211], [0.23554939031600952, 0.2325684279203415, 0.23559504747390747])
img_transform = standard_transforms.Compose([standard_transforms.ToTensor(), standard_transforms.Normalize(*mean_std)])
model_path = 'pretrained/pretrained_scale_prediction_model.pth'

def main(args):
    img_path = os.path.join(args.output_dir, 'images')
    dst_size_map_path = os.path.join(args.output_dir, 'size_map')
    if not os.path.exists(dst_size_map_path):
        os.makedirs(dst_size_map_path)
    file_list = os.listdir(img_path)
    GPU_ID = '0,1'
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    torch.backends.cudnn.benchmark = True
    net = CrowdCounter(GPU_ID, 'Res50_SCAR')
    net.cuda()
    net.load_state_dict(torch.load(model_path), strict=False)
    net.eval()
    gen_list = tqdm(file_list)
    for fname in gen_list:
        imgname = os.path.join(img_path, fname)
        size_map_path = os.path.join(dst_size_map_path, fname.split('.')[0] + '.jpg')
        if os.path.exists(size_map_path):
            continue
        else:
            img = Image.open(imgname)
            if img.mode == 'L':
                img = img.convert('RGB')
            img = img_transform(img)[None, :, :, :]
            with torch.no_grad():
                img = Variable(img).cuda()
                crop_imgs, crop_gt, crop_masks = [], [], []
                b, c, h, w = img.shape
                slice_h,slice_w = 768, 1024
                if h * w < slice_h * 2 * slice_w * 2 and h % 16 == 0 and w % 16 == 0:
                    pred_map = net.test_forward(img).cpu()
                else:
                    assert  h % 16 == 0 and w % 16 == 0
                    for i in range(0, h, slice_h):
                        h_start, h_end = max(min(h - slice_h, i), 0), min(h, i + slice_h)
                        for j in range(0, w, slice_w):
                            w_start, w_end = max(min(w - slice_w, j), 0), min(w, j + slice_w)
                            crop_imgs.append(img[:, :, h_start:h_end, w_start:w_end])
                            mask = torch.zeros(1,1,img.size(2), img.size(3)).cpu()
                            mask[:, :, h_start:h_end, w_start:w_end].fill_(1.0)
                            crop_masks.append(mask)
                    crop_imgs, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_imgs, crop_masks))
                    crop_preds = []
                    nz, period = crop_imgs.size(0), 8
                    for i in range(0, nz, period):
                        crop_pred = net.test_forward(crop_imgs[i:min(nz, i + period)]).cpu()
                        crop_preds.append(crop_pred)
                    crop_preds = torch.cat(crop_preds, dim=0)
                    idx = 0
                    pred_map = torch.zeros(1, 1, img.size(2), img.size(3)).cpu().float()
                    for i in range(0, h, slice_h):
                        h_start, h_end = max(min(h - slice_h, i), 0), min(h, i + slice_h)
                        for j in range(0, w, slice_w):
                            w_start, w_end = max(min(w - slice_w, j), 0), min(w, j + slice_w)
                            pred_map[:, :, h_start:h_end, w_start:w_end] += crop_preds[idx]
                            idx += 1
                    mask = crop_masks.sum(dim=0)
                    pred_map = (pred_map / mask)
                assert  pred_map[0, 0].size() == img[0,0].size()
                pred_map = pred_map.data.numpy()[0, 0, :, :]
            cv.imwrite(size_map_path, (pred_map * 255.0).astype(np.uint8), [cv.IMWRITE_JPEG_QUALITY, 100])