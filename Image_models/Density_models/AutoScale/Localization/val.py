import warnings
warnings.filterwarnings('ignore')
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import dataset
from find_contours import findmaxcontours
from fpn import AutoScale
import numpy as np
import cv2
from rate_model import RATEnet
from config import args
import os
import imageio
torch.cuda.manual_seed(args.seed)

def distance_generate(img_size, k, lamda, crop_size):
    distance = 1.0
    new_size = [0, 1]
    new_size[0] = img_size[2] * lamda
    new_size[1] = img_size[3] * lamda
    d_map = (np.zeros([int(new_size[0]), int(new_size[1])]) + 255).astype(np.uint8)
    gt = np.nonzero(k)
    if len(gt) == 0:
        distance_map = np.zeros([int(new_size[0]), int(new_size[1])])
        distance_map[:, :] = 10
        x = int(crop_size[0] * lamda)
        y = int(crop_size[1] * lamda)
        w = int(crop_size[2] * lamda)
        h = int(crop_size[3] * lamda)
        distance_map = distance_map[y:(y + h), x:(x + w)]
        return new_size, distance_map
    for o in range(0, len(gt)):
        x = int(max(1, gt[o][1].numpy() * lamda))
        y = int(max(1, gt[o][2].numpy() * lamda))
        if x >= new_size[0] - 1 or y >= new_size[1] - 1:
            continue
        d_map[x][y] = d_map[x][y] - 255
    distance_map = cv2.distanceTransform(d_map, cv2.DIST_L2, 5)
    distance_map[(distance_map >= 0) & (distance_map < 1)] = 0
    distance_map[(distance_map >= 1) & (distance_map < 2)] = 1
    distance_map[(distance_map >= 2) & (distance_map < 3)] = 2
    distance_map[(distance_map >= 3) & (distance_map < 4)] = 3
    distance_map[(distance_map >= 4) & (distance_map < 5 * distance)] = 4
    distance_map[(distance_map >= 5 * distance) & (distance_map < 6 * distance)] = 5
    distance_map[(distance_map >= 6 * distance) & (distance_map < 8 * distance)] = 6
    distance_map[(distance_map >= 8 * distance) & (distance_map < 12 * distance)] = 7
    distance_map[(distance_map >= 12 * distance) & (distance_map < 18 * distance)] = 8
    distance_map[(distance_map >= 18 * distance) & (distance_map < 28 * distance)] = 9
    distance_map[(distance_map >= 28 * distance)] = 10
    x = int(crop_size[0] * lamda)
    y = int(crop_size[1] * lamda)
    w = int(crop_size[2] * lamda)
    h = int(crop_size[3] * lamda)
    distance_map = distance_map[y:(y + h), x:(x + w)] # [225, 1035]
    return new_size, distance_map

def count_distance(input_img):
    input_img = input_img.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    imageio.imsave('distance_map.pgm', input_img)
    f = os.popen('./count_localminma/extract_local_minimum_return_xy ./distance_map.pgm 256 ./distance_map.pgm distance_map.pp')
    count = f.readlines()
    count = float(count[0].split('=')[1])
    return count

def validate(Pre_data, model, rate_model, args):
    # test loader
    test_loader = torch.utils.data.DataLoader(dataset.listDataset(Pre_data, transform=transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])), batch_size=args.batch_size)
    model.eval()
    mae = 0.0
    mse = 0.0
    for i, (fname, img, target, kpoint, sigma) in enumerate(test_loader):
        img_size = img.size()
        img = img.cuda() # [1, 3, 704, 1024]
        target = target.type(torch.LongTensor) # [1, 704, 1024]
        d0, d1, d2, d3, d4, d5, scales_feature = model(img, target, refine_flag=True) # [1, 11, 704, 1024], [1, 152, 704, 1024]
        original_distance_map = torch.max(F.softmax(d5), 1, keepdim=True)[1] # [1, 1, 704, 1024]
        crop_size = findmaxcontours(original_distance_map.data.cpu().numpy(), find_max=True, fname=fname)
        original_count = count_distance(original_distance_map)
        scale_crop = scales_feature[:, :, crop_size[1]:(crop_size[1] + crop_size[3]), crop_size[0]:(crop_size[0] + crop_size[2])] # [1, 152, 154, 707]
        scale_crop = F.adaptive_avg_pool2d(scale_crop, (14, 14)) # [1, 152, 14, 14]
        rate_feature = scale_crop
        rate_list = rate_model(rate_feature)
        rate_list.clamp_(0.5, 5)
        rate = torch.sqrt(rate_list)
        distance_map_gt_crop = distance_generate(img_size, kpoint, rate.item(), crop_size)[1] # [225, 1035]
        distance_map_gt_crop = torch.from_numpy(distance_map_gt_crop).unsqueeze(0).type(torch.LongTensor) # [1, 225, 1035]
        if (float(crop_size[2] * crop_size[3]) / (img_size[2] * img_size[3])) > args.area_threshold:
            img_crop = img[:, :, crop_size[1]:(crop_size[1] + crop_size[3]), crop_size[0]:(crop_size[0] + crop_size[2])]
            img_crop = F.upsample_bilinear(img_crop, (int(img_crop.size()[2] * rate), int(img_crop.size()[3] * rate)))
            dd0, dd1, dd2, dd3, dd4, dd5 = model(img_crop, distance_map_gt_crop, refine_flag=False)
            dd5 = torch.max(F.softmax(dd5), 1, keepdim=True)[1]
            count_crop = count_distance(dd5)
            original_distance_map[:, :, crop_size[1]:(crop_size[1] + crop_size[3]),
            crop_size[0]:(crop_size[0] + crop_size[2])] = 10
            count_other = count_distance(original_distance_map)
        else:
            count_crop = original_count
            count_other = 0
        count = count_crop + count_other
        Gt_count = torch.sum(kpoint).item()
        mae += abs(count - Gt_count)
        mse += abs(count - Gt_count) * abs(count - Gt_count)
        if i % args.print_freq == 0:
            print('File name: {}, Rate: {:.2f}, GT: {:.2f}, Pred: {:.2f}'.format(fname[0], rate.item(), torch.sum(kpoint).item(), count))
    mae = mae * 1.0 / len(test_loader)
    mse = math.sqrt(mse / len(test_loader))
    print('MAE: {:.2f}, MSE: {:.2f}'.format(mae, mse))

def main():
    if args.type_dataset == 'sha':
        test_file = 'npydata/ShanghaiA_test.npy'
    elif args.type_dataset == 'shb':
        test_file = 'npydata/ShanghaiB_test.npy'
    elif args.type_dataset == 'qnrf':
        test_file = 'npydata/qnrf_test.npy'
    elif args.type_dataset == 'jhu':
        test_file = 'npydata/jhu_val.npy'
    elif args.type_dataset == 'nwpu':
        test_file = 'npydata/nwpu_val_2048.npy'
    else:
        print('This dataset does not exist')
        raise NotImplementedError
    with open(test_file, 'rb') as outfile:
        val_list = np.load(outfile).tolist()
    # model
    model = AutoScale()
    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()
    rate_model = RATEnet()
    rate_model = nn.DataParallel(rate_model, device_ids=[0]).cuda()
    if args.pre:
        if os.path.isfile(args.pre):
            checkpoint = torch.load(args.pre, encoding='iso-8859-1')
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            rate_model.load_state_dict(checkpoint['rate_state_dict'])
            print('Load ckpt from:', args.pre)
        else:
            print('No ckpt found at:', args.pre)
    validate(val_list, model, rate_model, args)

if __name__ == '__main__':
    main()