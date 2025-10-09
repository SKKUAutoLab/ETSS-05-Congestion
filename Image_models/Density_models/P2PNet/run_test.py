import warnings
warnings.filterwarnings('ignore')
import argparse
from PIL import Image
import cv2
import numpy as np
import torch
import torchvision.transforms as standard_transforms
from models import build_model
import os

def main(args):
    # model
    model = build_model(args)
    model.cuda()
    # load trained model
    if args.ckpt_dir is not None:
        checkpoint = torch.load(args.ckpt_dir, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print('Load ckpt from:', args.ckpt_dir)
    model.eval()
    transform = standard_transforms.Compose([standard_transforms.ToTensor(), standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img_raw = Image.open(args.input_dir).convert('RGB')
    width, height = img_raw.size
    new_width = width // 128 * 128
    new_height = height // 128 * 128
    img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
    img = transform(img_raw)
    samples = torch.Tensor(img).unsqueeze(0)
    samples = samples.cuda() # [1, 3, 768, 1024]
    outputs = model(samples)
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0] # [49152]
    outputs_points = outputs['pred_points'][0] # [49152, 2]
    threshold = 0.5
    points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    predict_cnt = int((outputs_scores > threshold).sum())
    size = 2
    img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    for p in points:
        img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(args.output_dir, 'pred_{}.jpg'.format(predict_cnt)), img_to_draw)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--output_dir', type=str, default='saved_sha')
    parser.add_argument('--ckpt_dir', type=str, default='saved_sha/best_mae.pth')
    parser.add_argument('--input_dir', type=str, default='datasets/ShanghaiTech/part_A/train_data/images/IMG_1.jpg')
    # model config
    parser.add_argument('--backbone', default='vgg16_bn', type=str)
    parser.add_argument('--row', default=2, type=int)
    parser.add_argument('--line', default=2, type=int)
    args = parser.parse_args()

    print('Testing image:', args.input_dir)
    main(args)