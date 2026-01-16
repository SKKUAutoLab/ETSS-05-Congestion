import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as standard_transforms
import util.misc as utils
from models import build_model

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def visualization(samples, pred, vis_dir, img_path, split_map=None):
    pil_to_tensor = standard_transforms.ToTensor()
    restore_transform = standard_transforms.Compose([DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), standard_transforms.ToPILImage()])
    images = samples.tensors
    masks = samples.mask
    for idx in range(images.shape[0]):
        sample = restore_transform(images[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_vis = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()
        size = 3
        for p in pred[idx]:
            sample_vis = cv2.circle(sample_vis, (int(p[1]), int(p[0])), size, (0, 255, 0), -1)
        if split_map is not None:
            imgH, imgW = sample_vis.shape[:2]
            split_map = (split_map * 255).astype(np.uint8)
            split_map = cv2.applyColorMap(split_map, cv2.COLORMAP_JET)
            split_map = cv2.resize(split_map, (imgW, imgH), interpolation=cv2.INTER_NEAREST)
            sample_vis = split_map * 0.9 + sample_vis
        if vis_dir is not None:
            valid_area = torch.where(~masks[idx])
            valid_h, valid_w = valid_area[0][-1], valid_area[1][-1]
            sample_vis = sample_vis[:valid_h+1, :valid_w+1]
            name = img_path.split('/')[-1].split('.')[0]
            img_save_path = os.path.join(vis_dir, '{}_pred{}.jpg'.format(name, len(pred[idx])))
            cv2.imwrite(img_save_path, sample_vis)
            print('Saved image to:', img_save_path)

@torch.no_grad()
def evaluate_single_image(model, img_path, vis_dir=None):
    model.eval()
    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    transform = standard_transforms.Compose([standard_transforms.ToTensor(), standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img = transform(img)
    img = torch.Tensor(img)
    samples = utils.nested_tensor_from_tensor_list([img])
    samples = samples.to(torch.device('cuda'))
    img_h, img_w = samples.tensors.shape[-2:]
    outputs = model(samples, test=True)
    raw_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)
    outputs_scores = raw_scores[:, :, 1][0]
    outputs_points = outputs['pred_points'][0]
    print('Pred:', len(outputs_scores))
    if vis_dir: 
        points = [[point[0] * img_h, point[1] * img_w] for point in outputs_points]
        split_map = (outputs['split_map_raw'][0].detach().cpu().squeeze(0) > 0.5).float().numpy()
        visualization(samples, [points], vis_dir, img_path, split_map=split_map)

def main(args):
    # model
    model, criterion = build_model(args)
    model.cuda()
    # load trained model
    checkpoint = torch.load(args.resume, map_location='cpu')        
    model.load_state_dict(checkpoint['model'])
    # test
    vis_dir = None if args.vis_dir == "" else args.vis_dir
    evaluate_single_image(model, args.input_dir, vis_dir=vis_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--type_dataset', type=str, default='sha')
    parser.add_argument('--input_dir', type=str, default='data/ShanghaiTech/part_A/test_data/images/IMG_1.jpg')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='saved_sha/best_checkpoint.pth', type=str)
    parser.add_argument('--vis_dir', default="vis_sha", type=str)
    # model config
    parser.add_argument('--backbone', default='vgg16_bn', type=str)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned', 'fourier'))
    parser.add_argument('--dec_layers', default=2, type=int)
    parser.add_argument('--dim_feedforward', default=512, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    # loss config
    parser.add_argument('--set_cost_class', default=1, type=float)
    parser.add_argument('--set_cost_point', default=0.05, type=float)
    parser.add_argument('--ce_loss_coef', default=1.0, type=float)
    parser.add_argument('--point_loss_coef', default=5.0, type=float)
    parser.add_argument('--eos_coef', default=0.5, type=float)
    # testing config
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://')
    args = parser.parse_args()
    main(args)