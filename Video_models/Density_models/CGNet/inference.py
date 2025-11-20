import warnings
warnings.filterwarnings("ignore")
import argparse
import json
import os
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from torch.nn import SyncBatchNorm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from misc import tools
from models.tri_cropper import build_model
# from tri_dataset import build_video_dataset as build_dataset
from tri_dataset import build_video_dataset as build_dataset_sense
from tri_dataset_ht21 import build_video_dataset as build_dataset_ht21
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
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

def visualize_and_save(video_name, img_names, pos_lists, inflow_lists, vis_dir, input_dir):
    save_dir = os.path.join(vis_dir, video_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    counts = [len(p) for p in pos_lists]
    for frame_idx, (img_name, pos_list, inflow_list) in enumerate(zip(img_names, pos_lists, inflow_lists)):
        img_path = os.path.join(input_dir, video_name, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Cannot load image: {img_path}")
            continue
        for (x, y), inflow in zip(pos_list, inflow_list):
            color = (0, 255, 0) if inflow == 1 else (255, 0, 0)
            cv2.circle(img, (int(x), int(y)), 6, color, -1)
        density_map = generate_density_map(img.shape, pos_list)
        top_combined = np.concatenate((img, density_map), axis=1)
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
    img0 = cv2.imread(os.path.join(save_dir, "00000.jpg"))
    h, w, _ = img0.shape
    out_video = os.path.join(save_dir, f"{video_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, 15, (w, h))
    for i in range(len(pos_lists)):
        frame_path = os.path.join(save_dir, f"{i:05d}.jpg")
        frame = cv2.imread(frame_path)
        if frame is not None:
            writer.write(frame)
    writer.release()
    print(f"[INFO] Saved visualization video: {out_video}")
### draw density map and positions ###

def read_pts(path):
    with open(path, "r") as f:
        lines = f.readlines()
        pts = []
        for line in lines:
            line = line.strip().split(",")
            pts.append([float(line[0]), float(line[1])])
        pts = np.array(pts)
    return pts

def module2model(module_state_dict):
    state_dict = {}
    for k, v in module_state_dict.items():
        while k.startswith("module."):
            k = k[7:]
        if k == "n_averaged":
            print(f"{k}:{v}")
            continue
        state_dict[k] = v
    return state_dict

def main(pair_cfg, pair_ckpt, args):
    tools.init_distributed_mode(pair_cfg)
    tools.set_randomseed(42 + tools.get_rank())
    # model
    model = build_model(args=args)
    model.load_state_dict(module2model(torch.load(pair_ckpt)["model"]))
    model.cuda()
    print('Load ckpt from:', pair_ckpt)
    if pair_cfg.distributed:
        sync_model = SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(sync_model, device_ids=[pair_cfg.gpu], find_unused_parameters=False)
    # test loader
    if args.type_dataset == 'SENSE':
        dataset = build_dataset_sense(pair_cfg.Dataset.test.root, pair_cfg.Dataset.test.ann_dir)
    elif args.type_dataset == 'HT21':
        dataset = build_dataset_ht21(pair_cfg.Dataset.test.root)
    else:
        print('This dataset does not exist')
        raise NotImplementedError
    sampler = DistributedSampler(dataset, shuffle=False) if pair_cfg.distributed else None
    loader = DataLoader(dataset, batch_size=pair_cfg.Dataset.val.batch_size, sampler=sampler, shuffle=False, num_workers=pair_cfg.Dataset.val.num_workers, pin_memory=True)
    model.eval()
    video_results = {}
    with torch.no_grad():
        for imgs, labels in tqdm(loader):
            cnt_list = []
            video_name = labels["video_name"][0]
            img_names = labels["img_names"]
            img_name0 = img_names[0][0]
            if args.type_dataset == 'SENSE':
                pos_path0 = os.path.join("locater/results", video_name, img_name0 + ".txt")
            elif args.type_dataset == 'HT21':
                pos_path0 = os.path.join("locater/results", video_name, 'img1', img_name0 + ".txt")
            else:
                print('This dataset does not exist')
                raise NotImplementedError
            pos0 = read_pts(pos_path0)
            z0 = model.forward_single_image(imgs[0, 0].cuda().unsqueeze(0), [pos0], True)[0]
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
            pos_lists.append(pos0)
            inflow_lists.append([1 for _ in range(len(pos0))])
            memory_features = [[z0[i]] for i in range(len(pos0))]
            ttl_list = [args.ttl for _ in range(len(pos0))]
            for i in selected_idx:
                img_name = img_names[i][0]
                pos_path = os.path.join("locater/results", video_name, img_name + ".txt")
                pos = read_pts(pos_path)
                z = model.forward_single_image(imgs[0, i].cuda().unsqueeze(0), [pos], True)[0]
                z = F.normalize(z, dim=-1)
                C = np.zeros((len(pos), len(memory_features)))
                for idx, pre_z in enumerate(memory_features):
                    pre_z = torch.stack(pre_z[-1:], dim=0).unsqueeze(0)
                    pre_z = F.normalize(pre_z, dim=-1)
                    sim_cost = torch.bmm(pre_z, z.unsqueeze(0).transpose(1, 2))
                    sim_cost = sim_cost.cpu().numpy()[0]
                    sim_cost = np.min(sim_cost, axis=0)
                    C[:, idx] = sim_cost
                row_ind, col_ind = linear_sum_assignment(-C)
                sim_score = C[row_ind, col_ind]
                shared_mask = sim_score > args.threshold
                ori_shared_idx_list = col_ind[shared_mask]
                new_shared_idx_list = row_ind[shared_mask]
                outflow_idx_list = [i for i in range(len(pos)) if i not in row_ind[shared_mask]]
                for ori_idx, new_idx in zip(ori_shared_idx_list, new_shared_idx_list):
                    memory_features[ori_idx].append(z[new_idx])
                    ttl_list[ori_idx] = args.ttl
                for idx in outflow_idx_list:
                    memory_features.append([z[idx]])
                    ttl_list.append(args.ttl)
                pos_lists.append(pos)
                inflow_list = []
                for j in range(len(pos)):
                    if j in outflow_idx_list:
                        inflow_list.append(1)
                    else:
                        inflow_list.append(0)
                inflow_lists.append(inflow_list)
                cum_cnt += len(outflow_idx_list)
                cnt_list.append(len(outflow_idx_list))
                ttl_list = [ttl_list[idx] - 1 for idx in range(len(ttl_list))]
                for idx in range(len(ttl_list) - 1, -1, -1):
                    if ttl_list[idx] == 0:
                        del memory_features[idx]
                        del ttl_list[idx]
            pos_lists = [pos_lists[i].tolist() for i in range(len(pos_lists))]
            video_results[video_name] = {"video_num": cum_cnt, "first_frame_num": cnt_0, "cnt_list": cnt_list, "frame_num": len(img_names), "pos_lists": pos_lists, "inflow_lists": inflow_lists}
            ### draw density map and positions ###
            if args.is_vis:
                visualize_and_save(video_name=video_name, img_names=selected_img_names, pos_lists=video_results[video_name]["pos_lists"],
                                   inflow_lists=video_results[video_name]["inflow_lists"], vis_dir=args.vis_dir, input_dir=pair_cfg.Dataset.test.root)
            ### draw density map and positions ###
            print('Video name: {}, Number of video: {}, Count list: {}'.format(video_name, video_results[video_name]['video_num'], video_results[video_name]["cnt_list"]))
    with open(os.path.join(pair_cfg.Misc.tensorboard_dir, "video_results_test.json"), "w") as f:
        json.dump(video_results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--type_dataset', type=str, default='SENSE')
    parser.add_argument("--pair_config", default="configs/crowd_sense.json", type=str)
    parser.add_argument("--ckpt_dir", default="saved_sense/checkpoints/best.pth", type=str)
    parser.add_argument("--local_rank", type=int)
    # testing config
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--ttl', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--vis_dir', default="saved_sense/vis")
    parser.add_argument('--is_vis', action='store_true')
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    if os.path.exists(args.pair_config):
        with open(args.pair_config, "r") as f:
            pair_configs = json.load(f)
        pair_cfg = edict(pair_configs)
    main(pair_cfg, args.ckpt_dir, args)