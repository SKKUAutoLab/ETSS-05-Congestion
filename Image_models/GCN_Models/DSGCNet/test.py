import warnings
warnings.filterwarnings('ignore')
import argparse
import torchvision.transforms as standard_transforms
from PIL import Image
import cv2
import numpy as np
import torch
from models import build_model
import os

def main(args):
    # model
    model = build_model(args)
    model.cuda()
    if args.ckpt_dir is not None and os.path.exists(args.ckpt_dir):
        checkpoint = torch.load(args.ckpt_dir, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded ckpt from: {args.ckpt_dir}")
    else:
        print(f"[Warning] Weight file not found: {args.ckpt_dir}. Randomly initializing model.")
    model.eval()
    transform = standard_transforms.Compose([standard_transforms.ToTensor(), standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    scene_folders = os.listdir(os.path.join(args.input_dir, 'test_data/images'))
    scene_folders.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
    for scene_name in scene_folders:
        scene_path = os.path.join(args.input_dir, 'test_data/images', scene_name)
        if scene_path.endswith('.jpg'):
            img_raw = Image.open(scene_path).convert('RGB') # [704, 1024, 3]
            width, height = img_raw.size
            new_width = width // 128 * 128
            new_height = height // 128 * 128
            img_raw = img_raw.resize((new_width, new_height), cv2.INTER_CUBIC) # [1024, 640]
            img = transform(img_raw) # [3, 640, 1024]
            samples = torch.Tensor(img).unsqueeze(0).cuda() # [1, 3, 640, 1024]
            with torch.no_grad():
                outputs = model(samples)
            outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], dim=-1)[:, :, 1][0] # [40960]
            outputs_points = outputs['pred_points'][0] # [40960, 2]
            valid_mask = outputs_scores > args.threshold # [40960]
            points = outputs_points[valid_mask].detach().cpu().numpy().tolist()
            predict_cnt = int(valid_mask.sum().item())
            img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
            size = 3
            for p in points:
                cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
            text = str(predict_cnt)
            font_face = cv2.FONT_HERSHEY_TRIPLEX
            font_scale = 2.0
            thickness = 3
            color = (255, 255, 255)
            H, W, _ = img_to_draw.shape
            (text_w, text_h), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
            x_pos = W - text_w - 10
            y_pos = H - 10
            cv2.putText(img_to_draw, text, (x_pos, y_pos), font_face, font_scale, color, thickness)
            base_name = scene_name.split('.')[0]
            out_file_name = f"{base_name}.jpg"
            out_file_path = os.path.join(args.output_dir, out_file_name)
            cv2.imwrite(out_file_path, img_to_draw)
            print(f"Processed: {scene_name}, Predicted count: {predict_cnt}")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--type_dataset', type=str, default='sha', choices=['sha', 'shb'])
    parser.add_argument('--input_dir', type=str, default='datasets/ShanghaiTech/part_A')
    parser.add_argument('--output_dir', default='vis_sha', type=str)
    # model config
    parser.add_argument('--backbone', default='vgg16_bn', type=str)
    parser.add_argument('--row', default=2, type=int)
    parser.add_argument('--line', default=2, type=int)
    # testing config
    parser.add_argument('--ckpt_dir', default='saved_sha/latest.pth')
    parser.add_argument('--threshold', default=0.5, type=float)
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    main(args)