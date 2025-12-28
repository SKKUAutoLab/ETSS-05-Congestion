import warnings
warnings.filterwarnings('ignore')
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import numpy as np
import torch
from models import build_model
import os
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from test import read_pts as read_pts_sense
from test_ht21 import read_pts as read_pts_ht21
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

### draw density map and positions ###
def generate_density_map(img_shape, points, sigma=15):
    h, w = img_shape[:2]
    density = np.zeros((h, w), dtype=np.float32)
    for (x, y) in points:
        if 0 <= int(y) < h and 0 <= int(x) < w:
            density[int(y), int(x)] += 1
    density = cv2.GaussianBlur(density, (0, 0), sigma)
    density = density / (density.max() + 1e-8) * 255
    density = density.astype(np.uint8)
    density = cv2.applyColorMap(density, cv2.COLORMAP_JET)
    return density

def visualize_and_save(video_name, frames, img_names, pos_lists, inflow_lists, vis_dir, args):
    save_dir = os.path.join(vis_dir, video_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    counts = [len(p) for p in pos_lists]
    for frame_idx, (frame, img_name, pos_list, inflow_list) in enumerate(zip(frames, img_names, pos_lists, inflow_lists)):
        frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
        for (x, y), inflow in zip(pos_list, inflow_list):
            color = (0, 255, 0) if inflow == 1 else (255, 0, 0)
            cv2.circle(frame, (int(x), int(y)), 6, color, -1)
        density_map = generate_density_map(frame.shape, pos_list)
        top_combined = np.concatenate((frame, density_map), axis=1)
        dpi = 100
        fig_w = top_combined.shape[1] / dpi
        fig_h = 6
        plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        plt.plot(range(frame_idx + 1), counts[:frame_idx + 1], color="blue", linewidth=3)
        plt.xlabel("Frame", fontsize=30)
        plt.ylabel("Count", fontsize=30)
        ymax = max(max(counts) + 10, 250)
        plt.ylim(0, ymax)
        plt.yticks([0, 50, 100, 150, 200], fontsize=30)
        plt.xticks(fontsize=30)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"plot_{frame_idx:05d}.png")
        plt.savefig(plot_path)
        plt.close()
        plot_img = cv2.imread(plot_path)
        combined = np.concatenate((top_combined, plot_img), axis=0)
        cv2.putText(combined, f"Frame {frame_idx} | Count: {len(pos_list)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        save_path = os.path.join(save_dir, f"{frame_idx:05d}.jpg")
        cv2.imwrite(save_path, combined)
    out_video = os.path.join(save_dir, f"{video_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, 15, (args.width, args.height))
    for i in range(len(pos_lists)):
        frame_path = os.path.join(save_dir, f"{i:05d}.jpg")
        frame = cv2.imread(frame_path)
        if frame is not None:
            frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
            writer.write(frame)
    writer.release()
    print(f"[INFO] Saved visualization video: {out_video}")
### draw density map and positions ###

def video_transform(args):
    return A.Compose([A.Resize(args.height, args.width), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])

def main(args, demo_video_path='demo.mp4'):
    # model
    model, _ = build_model(args)
    model.cuda()
    # load trained model
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print('Load ckpt from:', args.resume)
    model.eval()
    # read video
    video_name = os.path.splitext(os.path.basename(demo_video_path))[0]
    if args.is_vis:
        save_dir = os.path.join(args.vis_dir, video_name)
        os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(demo_video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if args.height is None and args.width is None:
        args.height, args.width = frames[0].shape[:2]
    transform = video_transform(args)
    with torch.no_grad():
        cnt_list = []
        frame0 = frames[0]
        transformed = transform(image=frame0)['image']
        if args.type_dataset == 'SENSE':
            pos0, feature0 = read_pts_sense(model, transformed)
        elif args.type_dataset == 'HT21':
            pos0, feature0 = read_pts_ht21(model, transformed)
        else:
            print('This dataset does not exist')
            raise NotImplementedError
        pre_z = model.forward_single_image(transformed.cuda().unsqueeze(0), [pos0], feature0, True)
        pre_pos = pos0
        cnt_0 = len(pos0)
        cum_cnt = cnt_0
        cnt_list.append(cnt_0)
        selected_idx = [v for v in range(args.interval, len(frames), args.interval)]
        pos_lists = []
        inflow_lists = []
        outflow_lists = []
        pos_lists.append(pos0)
        inflow_lists.append([1 for _ in range(len(pos0))])
        for i in tqdm(selected_idx):
            frame_i = frames[i]
            transformed = transform(image=frame_i)['image']
            if args.type_dataset == 'SENSE':
                pos, feature1 = read_pts_sense(model, transformed)
            elif args.type_dataset == 'HT21':
                pos, feature1 = read_pts_ht21(model, transformed)
            else:
                print('This dataset does not exist')
                raise NotImplementedError
            pre_pre_z = pre_z
            z1, z2, pre_z = model.forward_single_image(transformed.cuda().unsqueeze(0), [pos], feature1, True, pre_z)
            z1 = F.normalize(z1, dim=-1).transpose(0, 1)
            z2 = F.normalize(z2, dim=-1).transpose(0, 1)
            sim_feats = torch.einsum('bnc, bmc->bnmc', z2, z1)
            sim_feats = sim_feats.view(1, -1, z1.shape[-1])
            pred_logits = model.vic.regression(sim_feats.squeeze(0))
            pred_probs = F.softmax(pred_logits, dim=1)
            pred_scores, pred_classes = pred_probs.max(dim=1)
            pedestrian_idx = torch.nonzero(pred_classes==0).squeeze(1).cpu().numpy()
            pedestrian_list = pedestrian_idx // z1.shape[1]
            pre_pedestrian_list = pedestrian_idx % z1.shape[1]
            inflow_idx_list = [i for i in range(len(pos)) if i not in pedestrian_list]
            outflow_idx_list = [i for i in range(len(pre_pos)) if i not in pre_pedestrian_list]
            pos_lists.append(pos)
            inflow_list = []
            for j in range(len(pos)):
                if j in inflow_idx_list:
                    inflow_list.append(1)
                else:
                    inflow_list.append(0)
            inflow_lists.append(inflow_list)
            cum_cnt += len(inflow_idx_list)
            cnt_list.append(len(inflow_idx_list))
            outflow_list = []
            for j in range(len(pre_pos)):
                if j in outflow_idx_list:
                    outflow_list.append(1)
                else:
                    outflow_list.append(0)
            outflow_lists.append(outflow_list)
            z_mask = np.array(outflow_list, dtype=bool)
            mem = pre_pre_z[0][:len(pre_pos)][z_mask]
            pre_z = [torch.cat((pre_z[0], mem), dim=0)]
            pre_pos = pos
        pos_lists = [pos_lists[i].tolist() for i in range(len(pos_lists))]
        ### draw density map and positions ###
        if args.is_vis:
            visualize_and_save(video_name=video_name, frames=frames, img_names=[f"{i:06d}.jpg" for i in range(len(frames))], pos_lists=pos_lists, inflow_lists=inflow_lists,
                               vis_dir=args.vis_dir, args=args)
        ### draw density map and positions ###
        print('Video name: {}, Count list: {}'.format(video_name, cnt_list))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--type_dataset', type=str, default='SENSE')
    parser.add_argument('--input_dir', type=str, default='demo.mp4')
    parser.add_argument('--resume', default='pretrained/SENSE.pth', type=str)
    parser.add_argument('--vis_dir', default="saved_results")
    parser.add_argument('--is_vis', action='store_true')
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('--width', type=int, default=None)
    # model config
    parser.add_argument('--backbone', default='convnext', type=str)
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
    args = parser.parse_args()

    print('Testing video:', args.input_dir)
    main(args)
