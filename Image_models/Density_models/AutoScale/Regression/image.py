import h5py
import numpy as np
from PIL import Image

def load_data(img_path, args):
    if args.type_dataset == 'qnrf':
        gt_path = img_path.replace('.jpg', '.h5').replace('.bmp', '.h5').replace('images', 'gt_distance_map')
    else:
        gt_path = img_path.replace('.jpg','.h5').replace('.bmp','.h5').replace('images', 'gt_density_map')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density_map'])
    k = np.asarray(gt_file['kpoint'])
    sigma_map = np.asarray(gt_file['sigma_map'])
    img = img.copy()
    target = target.copy()
    sigma_map = sigma_map.copy()
    k = k.copy()
    return img, target, k, sigma_map