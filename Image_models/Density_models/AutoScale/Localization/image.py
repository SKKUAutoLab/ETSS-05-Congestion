from PIL import Image
import numpy as np
import h5py

def load_data(img_path):
    gt_path = img_path.replace('.jpg','.h5').replace('images','gt_distance_map')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['distance_map'])
    kpoint = np.asarray(gt_file['kpoint'])
    sigma_map = np.asarray(gt_file['sigma_map'])
    img=img.copy()
    target=target.copy()
    sigma_map = sigma_map.copy()
    kpoint = kpoint.copy()
    return img, target, kpoint, sigma_map