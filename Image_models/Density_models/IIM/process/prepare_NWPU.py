import warnings
warnings.filterwarnings("ignore")
from PIL import  Image
import os
import cv2 as cv
import numpy as np
import json
from functions import euclidean_dist, generate_cycle_mask
import argparse
from natsort import natsorted

def generate_masks(args):
    all_imgs = natsorted(os.listdir(img_path))[:3609]
    for idx, img_id in enumerate(all_imgs):
        dst_mask_path = os.path.join(mask_path, img_id.replace('jpg', 'png'))
        if os.path.exists(dst_mask_path):
            continue
        else:
            ImgInfo = {}
            ImgInfo.update({"img_id": img_id})
            img_ori = os.path.join(img_path, img_id)
            img_ori = Image.open(img_ori)
            w, h = img_ori.size
            mask_map = np.zeros((h, w), dtype='uint8')
            gt_name = os.path.join(json_path, img_id.split('.')[0] + '.json')
            with open(gt_name) as f:
                ImgInfo = json.load(f)
            centroid_list = []
            wh_list = []
            for id,(w_start, h_start, w_end, h_end) in enumerate(ImgInfo["boxes"],0):
                centroid_list.append([(w_end + w_start) / 2, (h_end + h_start) / 2])
                wh_list.append([max((w_end - w_start) / 2, 3), max((h_end - h_start) / 2, 3)])
            centroids = np.array(centroid_list.copy(),dtype='int')
            wh = np.array(wh_list.copy(),dtype='int')
            wh[wh > 25] = 25
            human_num = ImgInfo["human_num"]
            for point in centroids:
                point = point[None, :]
                dists = euclidean_dist(point, centroids)
                dists = dists.squeeze()
                id = np.argsort(dists)
                for start, first in enumerate(id, 0):
                    if start > 0 and start < 5:
                        src_point = point.squeeze()
                        dst_point = centroids[first]
                        src_w, src_h = wh[id[0]][0], wh[id[0]][1]
                        dst_w, dst_h = wh[first][0], wh[first][1]
                        count = 0
                        if (src_w + dst_w) - np.abs(src_point[0] - dst_point[0]) > 0 and (src_h + dst_h) - np.abs(src_point[1] - dst_point[1]) > 0:
                            w_reduce = ((src_w + dst_w) - np.abs(src_point[0] - dst_point[0])) / 2
                            h_reduce = ((src_h + dst_h) - np.abs(src_point[1] - dst_point[1])) / 2
                            threshold_w, threshold_h = max(-int(max(src_w - w_reduce, dst_w - w_reduce) / 2.), -60), max(-int(max(src_h - h_reduce, dst_h - h_reduce) / 2.), -60)
                        else:
                            threshold_w, threshold_h = max(-int(max(src_w, dst_w) / 2.), -60), max(-int(max(src_h, dst_h) / 2.), -60)
                        while (src_w + dst_w) - np.abs(src_point[0] - dst_point[0]) > threshold_w and (src_h + dst_h) - np.abs(src_point[1] - dst_point[1]) > threshold_h:
                            if (dst_w * dst_h) > (src_w * src_h):
                                wh[first][0] = max(int(wh[first][0] * 0.9), 2)
                                wh[first][1] = max(int(wh[first][1] * 0.9), 2)
                                dst_w, dst_h = wh[first][0], wh[first][1]
                            else:
                                wh[id[0]][0] = max(int(wh[id[0]][0] * 0.9), 2)
                                wh[id[0]][1] = max(int(wh[id[0]][1] * 0.9), 2)
                                src_w, src_h = wh[id[0]][0], wh[id[0]][1]
                            if human_num >= 3:
                                dst_point_ = centroids[id[start + 1]]
                                dst_w_, dst_h_ = wh[id[start + 1]][0], wh[id[start + 1]][1]
                                if (dst_w_ * dst_h_) > (src_w * src_h) and (dst_w_ * dst_h_) > (dst_w * dst_h):
                                    if (src_w + dst_w_) - np.abs(src_point[0] - dst_point_[0]) > -3 and (src_h + dst_h_) - np.abs(src_point[1] - dst_point_[1]) > -3:
                                        wh[id[start + 1]][0] = max(int(wh[id[start + 1]][0] * 0.9), 2)
                                        wh[id[start + 1]][1] = max(int(wh[id[start + 1]][1] * 0.9), 2)
                            count += 1
                            if count > 40:
                                break
            for (center_w, center_h), (width, height) in zip(centroids, wh):
                assert (width > 0 and height > 0)
                if (0 < center_w < w) and (0 < center_h < h):
                    h_start = (center_h - height)
                    h_end = (center_h + height)
                    w_start = center_w - width
                    w_end = center_w + width
                    if h_start < 0:
                        h_start = 0
                    if h_end > h:
                        h_end = h
                    if w_start < 0:
                        w_start = 0
                    if w_end > w:
                        w_end = w
                    if args.cycle:
                        mask = generate_cycle_mask(height, width)
                        mask_map[h_start:h_end, w_start: w_end] = mask
                    else:
                        mask_map[h_start:h_end, w_start: w_end] = 1
            mask_map = mask_map * 255
            cv.imwrite(dst_mask_path, mask_map, [cv.IMWRITE_PNG_BILEVEL, 1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='NWPU')
    parser.add_argument('--input_dir', type=str, default='data/NWPU-Crowd')
    parser.add_argument('--cycle', type=bool, default=False)
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    img_path = os.path.join(args.input_dir, 'images')
    json_path = os.path.join(args.input_dir, 'jsons')
    mask_path = os.path.join(args.input_dir, 'mask_50_60')
    if  not os.path.exists(mask_path):
        os.makedirs(mask_path)
    generate_masks(args)