import warnings
warnings.filterwarnings('ignore')
import random
import json
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from datasets import build_dataset
from models import build_model
import util.misc as utils
from util.misc import nested_tensor_from_tensor_list
import os
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

# def visualize_and_save(video_name, img_names, pos_lists, inflow_lists, vis_dir, input_dir):
#     save_dir = os.path.join(vis_dir, video_name)
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     counts = [len(p) for p in pos_lists]
#     for frame_idx, (img_name, pos_list, inflow_list) in enumerate(zip(img_names, pos_lists, inflow_lists)):
#         img_path = os.path.join(input_dir, video_name, 'img1', img_name)
#         img = cv2.imread(img_path)
#         if img is None:
#             print(f"[WARN] Cannot load image: {img_path}")
#             continue
#         img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR) # avoid out of ram issue
#         for (x, y), inflow in zip(pos_list, inflow_list):
#             color = (0, 255, 0) if inflow == 1 else (255, 0, 0)
#             cv2.circle(img, (int(x), int(y)), 6, color, -1)
#         density_map = generate_density_map(img.shape, pos_list)
#         top_combined = np.concatenate((img, density_map), axis=1)
#         dpi = 100
#         fig_w = top_combined.shape[1] / dpi
#         fig_h = 6
#         plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
#         plt.plot(range(frame_idx + 1), counts[:frame_idx + 1], color="blue", linewidth=3)
#         plt.xlabel("Frame", fontsize=30)
#         plt.ylabel("Count", fontsize=30)
#         ymax = max(max(counts) + 10, 250)
#         plt.ylim(0, ymax)
#         plt.yticks([0, 50, 100, 150, 200], fontsize=30)
#         plt.xticks(fontsize=30)
#         plt.grid(True, linestyle="--", alpha=0.5)
#         plt.tight_layout()
#         plot_path = os.path.join(save_dir, f"plot_{frame_idx:05d}.png")
#         plt.savefig(plot_path)
#         plt.close()
#         plot_img = cv2.imread(plot_path)
#         combined = np.concatenate((top_combined, plot_img), axis=0)
#         cv2.putText(combined, f"Frame {frame_idx} | Count: {len(pos_list)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#         save_path = os.path.join(save_dir, f"{frame_idx:05d}.jpg")
#         cv2.imwrite(save_path, combined)
#     img0 = cv2.imread(os.path.join(save_dir, "00000.jpg"))
#     h, w, _ = img0.shape
#     out_video = os.path.join(save_dir, f"{video_name}.mp4")
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     writer = cv2.VideoWriter(out_video, fourcc, 15, (w, h))
#     for i in range(len(pos_lists)):
#         frame_path = os.path.join(save_dir, f"{i:05d}.jpg")
#         frame = cv2.imread(frame_path)
#         if frame is not None:
#             writer.write(frame)
#     writer.release()
#     print(f"[INFO] Saved visualization video: {out_video}")

def visualize_and_save(video_name, img_names, pos_lists, inflow_lists, vis_dir, input_dir):
    save_dir = os.path.join(vis_dir, video_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    counts = [len(p) for p in pos_lists]
    total_frames = len(img_names)
    max_count = max(counts) if counts else 1
    step = max(1, max_count // 5)
    yticks = [i for i in range(0, max_count + 1, step)]
    for frame_idx, (img_name, pos_list, inflow_list) in enumerate(zip(img_names, pos_lists, inflow_lists)):
        img_path = os.path.join(input_dir, video_name, 'img1', img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Cannot load image: {img_path}")
            continue
        img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR)
        for (x, y), inflow in zip(pos_list, inflow_list):
            color = (0, 255, 0) if inflow == 1 else (255, 0, 0)
            cv2.circle(img, (int(x), int(y)), 6, color, -1)
        density_map = generate_density_map(img.shape, pos_list)
        top_combined = np.concatenate((img, density_map), axis=1)
        dpi = 100
        fig_w = top_combined.shape[1] / dpi
        fig_h = 6
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        ax.axhspan(0, 24, facecolor="green", alpha=0.15, label="Low Congestion (0–24)")
        ax.axhspan(25, 100, facecolor="yellow", alpha=0.15, label="Medium Congestion (25–100)")
        ax.axhspan(101, max_count, facecolor="red", alpha=0.15, label="High Congestion (> 100)")
        ax.plot(range(frame_idx + 1), counts[:frame_idx + 1], color="black", linewidth=3)
        ax.set_xlim(0, total_frames)
        ax.set_xticks(np.arange(0, total_frames + 1, total_frames // 5 if total_frames > 20 else 1))
        ax.set_ylim(0, max_count)
        ax.set_yticks(yticks)
        ax.set_xlabel("Frame", fontsize=30)
        ax.set_ylabel("Count", fontsize=30)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=20)
        ax.tick_params(axis='x', labelsize=25)
        ax.tick_params(axis='y', labelsize=25)
        fig.tight_layout()
        plot_path = os.path.join(save_dir, f"plot_{frame_idx:05d}.png")
        plt.savefig(plot_path)
        plt.close(fig)
        plot_img = cv2.imread(plot_path)
        if plot_img.shape[1] != top_combined.shape[1]:
            plot_img = cv2.resize(plot_img, (top_combined.shape[1], plot_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        combined = np.concatenate((top_combined, plot_img), axis=0)
        cv2.putText(combined, f"Frame {frame_idx} | Count: {len(pos_list)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        save_path = os.path.join(save_dir, f"{frame_idx:05d}.jpg")
        cv2.imwrite(save_path, combined)
    img0 = cv2.imread(os.path.join(save_dir, "00000.jpg"))
    h, w, _ = img0.shape
    out_video = os.path.join(save_dir, f"{video_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, 5, (w, h))
    for i in range(len(pos_lists)):
        frame_path = os.path.join(save_dir, f"{i:05d}.jpg")
        frame = cv2.imread(frame_path)
        if frame is not None:
            writer.write(frame)
    writer.release()
    print(f"[INFO] Saved visualization video: {out_video}")
### draw density map and positions ###

def setup_seed(args):
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def read_pts(model, img):
    if isinstance(img, (list, torch.Tensor)):
        samples = nested_tensor_from_tensor_list(img.unsqueeze(0).cuda())
    outputs, features = model(samples, [], [], test=True)
    outputs_points = outputs['pred_points'][0].cpu()
    outputs_points = outputs_points.detach().numpy()
    img_h, img_w = samples.tensors.shape[-2:]
    points = [[float(point[1] * img_w), float(point[0] * img_h)] for point in outputs_points]
    if len(points) == 0:
        points.append([1, 1])
    return np.array(points, dtype ='float32'), features['4x'].tensors

def main(args):
    utils.init_distributed_mode(args)
    setup_seed(args)
    # model
    model, criterion = build_model(args)
    model.cuda()
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    # test loader
    dataset_test = build_dataset(args.type_dataset, args.input_dir, step=args.step)
    sampler_val = DistributedSampler(dataset_test, shuffle=False) if args.distributed else None
    data_loader_val = DataLoader(dataset_test, batch_size=args.batch_size, sampler=sampler_val, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    # load pretrained model
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print('Load ckpt from:', args.resume)
    model.eval()
    video_results = {}
    with torch.no_grad():
        for imgs, labels in tqdm(data_loader_val):
            cnt_list = []
            video_name = labels["video_name"][0]
            img_names = labels["img_names"]
            pos0, feature0 = read_pts(model, imgs[0, 0])
            z0 = model.forward_single_image(imgs[0, 0].cuda().unsqueeze(0), [pos0], feature0, True)
            pre_z = z0
            pre_pos = pos0
            cnt_0 = len(pos0)
            cum_cnt = cnt_0
            cnt_list.append(cnt_0)
            selected_idx = [v for v in range(args.interval, len(img_names), args.interval)]
            ### draw density map and positions ###
            if args.is_vis:
                draw_selected_idx = [0] + [v for v in range(args.interval, len(img_names), args.interval)]
                selected_img_names = [img_names[i][0] for i in draw_selected_idx]
            ### draw density map and positions ###
            pos_lists = []
            inflow_lists = []
            outflow_lists = []
            pos_lists.append(pos0)
            inflow_lists.append([1 for _ in range(len(pos0))])
            for i in selected_idx:
                pos, feature1 = read_pts(model, imgs[0, i])
                pre_pre_z = pre_z
                z1, z2, pre_z = model.forward_single_image(imgs[0, i].cuda().unsqueeze(0), [pos], feature1, True, pre_z)
                z1 = F.normalize(z1, dim=-1).transpose(0, 1)
                z2 = F.normalize(z2, dim=-1).transpose(0, 1)
                match_matrix = torch.bmm(z1, z2.transpose(1, 2))
                C = match_matrix.cpu().detach().numpy()[0]
                row_ind, col_ind = linear_sum_assignment(-C)
                sim_feat = z1[:, row_ind, :] * z2[:, col_ind, :]
                pred_logits = model.vic.regression(sim_feat.squeeze(0))
                pred_prob = F.softmax(pred_logits, dim=1)
                pred_score, pred_class = pred_prob.max(dim=1)
                pedestrian_list = col_ind[(1 - pred_class).bool().cpu().numpy()]
                pre_pedestrian_list = row_ind[(1 - pred_class).bool().cpu().numpy()]
                inflow_idx_list = [i for i in range(len(pos)) if i not in pedestrian_list]
                outflow_idx_list = [i for i in range(len(pre_pos)) if i not in pre_pedestrian_list]
                if video_name in ['HT21-15', 'HT21-13', 'HT21-12']: # double check due to many inflows
                    if inflow_idx_list:
                        for idx2 in inflow_idx_list:
                            for idx1 in range(z1.shape[1]):
                                sim_feat2 = z1[:, idx1, :] * z2[:, idx2, :]
                                pred_logits2 = model.vic.regression(sim_feat2.squeeze(0))
                                pred_prob2 = F.softmax(pred_logits2, dim=0)
                                pred_score2, pred_class2 = pred_prob2.max(dim=0)
                                if pred_class2 == 0:
                                    pedestrian_list = np.append(pedestrian_list, idx2)
                                    pre_pedestrian_list = np.append(pre_pedestrian_list, idx1)
                                    break
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
                if video_name == 'HT21-11':
                    z_mask = np.array (outflow_list, dtype = bool)
                    pre_z = [torch.cat((pre_z[0], pre_pre_z[0][:len(pre_pos)][z_mask]),dim=0)]
                pre_pos = pos
            pos_lists = [pos_lists[i].tolist() for i in range(len(pos_lists))]
            video_results[video_name] = {"video_num": cum_cnt, "first_frame_num": cnt_0, "cnt_list": cnt_list, "frame_num": len(img_names), "pos_lists": pos_lists, "inflow_lists": inflow_lists}
            ### draw density map and positions ###
            if args.is_vis:
                visualize_and_save(video_name=video_name, img_names=selected_img_names, pos_lists=video_results[video_name]["pos_lists"],
                                   inflow_lists=video_results[video_name]["inflow_lists"], vis_dir=args.vis_dir, input_dir=args.input_dir)
            ### draw density map and positions ###
            print('Video name: {}, Number of video: {}, Count list: {}'.format(video_name, video_results[video_name]['video_num'], video_results[video_name]["cnt_list"]))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, "video_results_test.json"), "w") as f:
        json.dump(video_results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--type_dataset', type=str, default='HT21')
    parser.add_argument('--input_dir', type=str, default='data/HT21/test')
    parser.add_argument('--max_len', default=3000)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='pretrained/HT21.pth', type=str)
    parser.add_argument('--vis_dir', default="saved_ht21/vis")
    parser.add_argument('--output_dir', type=str, default='saved_ht21')
    parser.add_argument('--is_vis', action='store_true')
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
    # testing config
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--interval', type=int, default=75)
    parser.add_argument('--step', type=int, default=15)
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    main(args)