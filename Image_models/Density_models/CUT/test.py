import argparse
import torch
from torch.utils.data import DataLoader
from dataset.dataset import Crowd
from model.model import Count
import numpy as np
torch.backends.cudnn.benchmark = True

def main(args):
    # test loader
    dataset = Crowd(args.input_dir, args.crop_size, args.downsample_ratio, method='val')
    dataloader = DataLoader(dataset, args.batch_size, shuffle=False, pin_memory=False)
    # model
    model = Count(args)
    model.cuda()
    model.load_state_dict(torch.load(args.ckpt_dir, map_location='cuda'))
    model.eval()
    print('Load ckpt from:', args.ckpt_dir)
    res = []
    step = 0
    for im, gt, size in dataloader:
        im = im.cuda()
        with torch.set_grad_enabled(False):
            result, _, _, _, _, _ = model(im)
            res1 = gt.item() - torch.sum(result).item()
            res.append(res1)
            print('Step: [{}/{}], GT: {}, Pred: {:.4f}, Diff: {:.4f}'.format(step, len(dataset), gt.item(), torch.sum(result).item(), res1), size[0])
            step = step + 1
    print('MAE: {:.4f}, MSE:{:.4f}'.format(np.mean(np.abs(res)), np.sqrt(np.mean(np.square(res)))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--type_dataset', type=str, default='sha')
    parser.add_argument('--input_dir', default='processed_datasets/sha/val', type=str)
    parser.add_argument('--crop_size', default=512, type=int)
    parser.add_argument('--batch_size', type=int, default=1)
    # model config
    parser.add_argument('--downsample_ratio', default=8, type=int)
    parser.add_argument('--pretrained', default='weights/pcpvt_large.pth')
    parser.add_argument('--ckpt_dir', default='saved_sha/best_model_32.pth')
    parser.add_argument('--drop', type=float, default=0.0)
    parser.add_argument('--drop_path', type=float, default=0.45)
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    main(args)