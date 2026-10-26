import os
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import find_objects
from skimage.segmentation import slic
from tqdm import tqdm


def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w * ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w * ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h * ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h * ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio


def filter_points(points, im_w, im_h):
    if len(points) == 0:
        return points.reshape(0, 2)
    keep = (points[:, 0] >= 0) & (points[:, 0] <= im_w) & (points[:, 1] >= 0) & (points[:, 1] <= im_h)
    return points[keep]


def resize_sample(img, points, min_size, max_size):
    im_w, im_h = img.size
    points = filter_points(points, im_w, im_h)
    new_h, new_w, ratio = cal_new_size(im_h, im_w, min_size, max_size)
    if ratio != 1.0:
        img = Image.fromarray(cv2.resize(np.array(img), (new_w, new_h), interpolation=cv2.INTER_CUBIC))
        points = points * ratio
    return img, points


def read_sha_points(path):
    return loadmat(path)["image_info"][0, 0][0, 0][0].astype(np.float32).reshape(-1, 2)


def read_ucf_points(path):
    return loadmat(path)["annPoints"].astype(np.float32).reshape(-1, 2)


def read_jhu_points(path):
    points = []
    with open(path) as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) < 2:
                continue
            points.append([float(vals[0]), float(vals[1])])
    return np.array(points, dtype=np.float32).reshape(-1, 2)


def nnec_radius(points, im_h, im_w):
    n = points.shape[0]
    if n == 1:
        return np.array([float(max(im_h, im_w))], dtype=np.float32)
    pts = torch.from_numpy(points).float()
    dist = torch.cdist(pts, pts)
    dist.fill_diagonal_(float("inf"))
    return dist.min(dim=1)[0].numpy()


class EDPSAM:
    def __init__(self, checkpoint, model_type, device, batch_size, n_segments):
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
        sam.eval()
        self.predictor = SamPredictor(sam)
        self.device = device
        self.batch_size = batch_size
        self.n_segments = n_segments

    def superpixel_boxes(self, image, points):
        im_h, im_w = image.shape[:2]
        seg = slic(image, n_segments=self.n_segments, start_label=1)
        slices = find_objects(seg)
        xs = np.clip(points[:, 0].astype(np.int64), 0, im_w - 1)
        ys = np.clip(points[:, 1].astype(np.int64), 0, im_h - 1)
        boxes = np.zeros((points.shape[0], 4), dtype=np.float32)
        for i, (x, y) in enumerate(zip(xs, ys)):
            sl = slices[seg[y, x] - 1]
            boxes[i] = [sl[1].start, sl[0].start, sl[1].stop - 1, sl[0].stop - 1]
        return boxes

    @torch.no_grad()
    def __call__(self, img, points):
        image = np.array(img.convert("RGB"))
        im_h, im_w = image.shape[:2]
        instance = np.zeros((im_h, im_w), dtype=np.int32)
        n = points.shape[0]
        if n == 0:
            return instance
        boxes = self.superpixel_boxes(image, points)
        radii = nnec_radius(points, im_h, im_w)
        self.predictor.set_image(image)
        pieces = []
        for s in range(0, n, self.batch_size):
            e = min(n, s + self.batch_size)
            coords = torch.from_numpy(points[s:e]).float().to(self.device)[:, None, :]
            labels = torch.ones((e - s, 1), dtype=torch.int64, device=self.device)
            bxs = torch.from_numpy(boxes[s:e]).float().to(self.device)
            coords = self.predictor.transform.apply_coords_torch(coords, (im_h, im_w))
            bxs = self.predictor.transform.apply_boxes_torch(bxs, (im_h, im_w))
            masks, _, _ = self.predictor.predict_torch(
                point_coords=coords,
                point_labels=labels,
                boxes=bxs,
                multimask_output=False,
            )
            for k in range(e - s):
                idx = s + k
                cx, cy, r = float(points[idx, 0]), float(points[idx, 1]), float(radii[idx])
                x0, x1 = max(0, int(np.floor(cx - r))), min(im_w, int(np.ceil(cx + r)) + 1)
                y0, y1 = max(0, int(np.floor(cy - r))), min(im_h, int(np.ceil(cy + r)) + 1)
                if x1 <= x0 or y1 <= y0:
                    continue
                yy = torch.arange(y0, y1, device=self.device).float()[:, None]
                xx = torch.arange(x0, x1, device=self.device).float()[None, :]
                circle = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
                inst = (masks[k, 0, y0:y1, x0:x1] & circle).cpu().numpy()
                area = int(inst.sum())
                if area == 0:
                    continue
                pieces.append((area, idx, y0, x0, inst))
        for area, idx, y0, x0, inst in sorted(pieces, key=lambda p: -p[0]):
            h, w = inst.shape
            instance[y0:y0 + h, x0:x0 + w][inst] = idx + 1
        return instance


def save_sample(out_dir, img_name, anno_name, img, points, src_img_path, mask_fn):
    img_dir = os.path.join(out_dir, "images")
    anno_dir = os.path.join(out_dir, "new-anno")
    mask_dir = os.path.join(out_dir, "masks")
    for d in (img_dir, anno_dir, mask_dir):
        os.makedirs(d, exist_ok=True)
    img_path = os.path.join(img_dir, img_name + ".jpg")
    if src_img_path is None or os.path.abspath(src_img_path) != os.path.abspath(img_path):
        img.save(img_path, quality=95)
    np.save(os.path.join(anno_dir, anno_name), points.astype(np.float32).reshape(-1, 2))
    if mask_fn is not None:
        np.save(os.path.join(mask_dir, img_name + ".npy"), mask_fn(img, points))


def process_sha(root, mask_fn):
    for split in ("train_data", "test_data"):
        split_dir = os.path.join(root, split)
        img_dir = os.path.join(split_dir, "images")
        names = sorted(f[:-4] for f in os.listdir(img_dir) if f.endswith(".jpg"))
        for name in tqdm(names, desc=f"SHA {split}"):
            src = os.path.join(img_dir, name + ".jpg")
            img = Image.open(src)
            points = read_sha_points(os.path.join(split_dir, "ground-truth", f"GT_{name}.mat"))
            save_sample(split_dir, name, f"GT_{name}.npy", img, points, src, mask_fn)


def process_ucf(root, mask_fn, min_size, max_size):
    for src_split, dst_split in (("Train", "train_data"), ("Test", "test_data")):
        src_dir = os.path.join(root, src_split)
        dst_dir = os.path.join(root, dst_split)
        names = sorted(f[:-4] for f in os.listdir(src_dir) if f.endswith(".jpg"))
        for name in tqdm(names, desc=f"UCF {src_split}"):
            img = Image.open(os.path.join(src_dir, name + ".jpg")).convert("RGB")
            points = read_ucf_points(os.path.join(src_dir, f"{name}_ann.mat"))
            img, points = resize_sample(img, points, min_size, max_size)
            save_sample(dst_dir, name, f"{name}_ann.npy", img, points, None, mask_fn)


def process_jhu(root, mask_fn, min_size, max_size):
    for src_split, dst_split in (("train", "train_data"), ("test", "test_data")):
        src_dir = os.path.join(root, src_split)
        dst_dir = os.path.join(root, dst_split)
        img_dir = os.path.join(src_dir, "images")
        names = sorted(f[:-4] for f in os.listdir(img_dir) if f.endswith(".jpg"))
        for i, name in enumerate(tqdm(names, desc=f"JHU {src_split}"), start=1):
            new_name = f"{i:04d}"
            img = Image.open(os.path.join(img_dir, name + ".jpg")).convert("RGB")
            points = read_jhu_points(os.path.join(src_dir, "gt", name + ".txt"))
            img, points = resize_sample(img, points, min_size, max_size)
            save_sample(dst_dir, new_name, f"{new_name}.npy", img, points, None, mask_fn)


def parse_args():
    p = argparse.ArgumentParser("Process SHA / UCF-QNRF / JHU-CROWD++ for XMask")
    p.add_argument("--root", default="datasets")
    p.add_argument("--datasets", nargs="+", default=["sha", "ucf", "jhu"], choices=["sha", "ucf", "jhu"])
    p.add_argument("--min-size", type=int, default=512)
    p.add_argument("--ucf-max-size", type=int, default=1536)
    p.add_argument("--jhu-max-size", type=int, default=2048)
    p.add_argument("--sam-checkpoint", default="checkpoints/sam_vit_h_4b8939.pth")
    p.add_argument("--sam-type", default="vit_h")
    p.add_argument("--sam-batch", type=int, default=32)
    p.add_argument("--n-segments", type=int, default=1000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    mask_fn = EDPSAM(args.sam_checkpoint, args.sam_type, args.device, args.sam_batch, args.n_segments)
    if "sha" in args.datasets:
        process_sha(os.path.join(args.root, "SHA"), mask_fn)
    if "ucf" in args.datasets:
        process_ucf(os.path.join(args.root, "UCF"), mask_fn, args.min_size, args.ucf_max_size)
    if "jhu" in args.datasets:
        process_jhu(os.path.join(args.root, "JHU"), mask_fn, args.min_size, args.jhu_max_size)


if __name__ == "__main__":
    main()
