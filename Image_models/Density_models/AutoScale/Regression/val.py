import warnings
warnings.filterwarnings('ignore')
import math
import torch
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter
from torchvision import transforms
import scipy
import dataset
from find_couter import findmaxcontours
from fpn import AutoScale
import numpy as np
from rate_model import RATEnet
from config import args
import  os
torch.cuda.manual_seed(args.seed)

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

def target_transform(gt_point, rate):
    point_map = gt_point.cpu().numpy()
    pts = np.array(list(zip(np.nonzero(point_map)[2], np.nonzero(point_map)[1])))
    pt2d = np.zeros((int(rate * point_map.shape[1]) + 1, int(rate * point_map.shape[2]) + 1), dtype=np.float32)
    for i, pt in enumerate(pts):
        pt2d[int(rate * pt[1]), int(rate * pt[0])] = 1.0
    return pt2d

def gt_transform(pt2d, cropsize, rate):
    [x, y, w, h] = cropsize
    pt2d = pt2d[int(y * rate):int(rate * (y + h)), int(x * rate):int(rate * (x + w))]
    density = np.zeros((int(pt2d.shape[0]), int(pt2d.shape[1])), dtype=np.float32)
    pts = np.array(list(zip(np.nonzero(pt2d)[1], np.nonzero(pt2d)[0])))
    orig = np.zeros((int(pt2d.shape[0]), int(pt2d.shape[1])), dtype=np.float32)
    for i, pt in enumerate(pts):
        orig[int(pt[1]), int(pt[0])] = 1.0
    density += scipy.ndimage.filters.gaussian_filter(orig, 4, mode='constant')
    return density

def validate(Pre_data, model, rate_model, args):
    test_loader = torch.utils.data.DataLoader(dataset.listDataset(Pre_data, args=args, transform=transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), train=False), batch_size=args.batch_size)
    model.eval()
    mae = 0
    mse = 0
    for i, (img, target, kpoint, fname, sigma_map) in enumerate(test_loader):
        img = img.cuda()
        target = target.cuda()
        d2, d3, d4, d5, d6, fs = model(img, target, refine_flag=True)
        density_map = d6.data.cpu().numpy()
        original_density = d6
        [x, y, w, h] = findmaxcontours(density_map, fname)
        rate_feature = F.adaptive_avg_pool2d(fs[:, :, y:(y + h), x:(x + w)], (14, 14))
        rate = rate_model(rate_feature).clamp_(0.5, 9)
        rate = torch.sqrt(rate)
        if (float(w * h) / (img.size(2) * img.size(3))) > args.area_threshold:
            img_pros = img[:, :, y:(y + h), x:(x + w)]
            img_transed = F.upsample_bilinear(img_pros, scale_factor=rate.item())
            pt2d = target_transform(kpoint, rate)
            target_choose = gt_transform(pt2d, [x, y, w, h], rate.item())
            target_choose = torch.from_numpy(target_choose).type(torch.FloatTensor).unsqueeze(0)
            dd2, dd3, dd4, dd5, dd6 = model(img_transed, target_choose, refine_flag=False)
            temp = dd6.data.cpu().numpy().sum()
            original_density[:, :, y:(y + h), x:(x + w)] = 0
            count = original_density.data.cpu().numpy().sum() + temp
        else:
            count = d6.data.cpu().numpy().sum()
        mae += abs(count - target.data.cpu().numpy().sum())
        mse += abs(count - target.data.cpu().numpy().sum()) * abs(count - target.data.cpu().numpy().sum())
        if i % args.print_freq == 0:
            print('File name: {}, Rate: {:.2f}, GT: {:.2f}, Pred: {:.2f}'.format(fname[0], rate.item(), target.data.cpu().numpy().sum(), count))
    mae = mae / len(test_loader)
    mse = math.sqrt(mse/len(test_loader))
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
        test_file = 'npydata/nwpu_val_1024.npy'
    else:
        print('This dataset does not exist')
        raise NotImplementedError
    with open(test_file, 'rb') as outfile:
        val_list = np.load(outfile).tolist()
    # model
    model = AutoScale().cuda()
    rate_model = RATEnet().cuda()
    if args.pre:
        if os.path.isfile(args.pre):
            checkpoint = torch.load(args.pre, encoding='iso-8859-1')
            model_state_dict = remove_module_prefix(checkpoint['state_dict'])
            rate_model_state_dict = remove_module_prefix(checkpoint['rate_state_dict'])
            model.load_state_dict(model_state_dict, strict=False)
            rate_model.load_state_dict(rate_model_state_dict)
            print('Load ckpt from:', args.pre)
        else:
            print('No ckpt found at:', args.pre)
    validate(val_list, model, rate_model, args)

if __name__ == '__main__':
    main()