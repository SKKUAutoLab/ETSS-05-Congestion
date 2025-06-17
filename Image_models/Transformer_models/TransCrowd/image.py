from PIL import Image
import numpy as np
import h5py
import cv2

def load_data(img_path):
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'gt_density_map')
    img = Image.open(img_path).convert('RGB')
    while True:
        try:
            gt_file = h5py.File(gt_path)
            gt_count = np.asarray(gt_file['gt_count'])
            break
        except OSError:
            print("Load img error:", img_path)
            cv2.waitKey(1000)
    img = img.copy() # [384, 384, 3]
    gt_count = gt_count.copy() # [1]
    return img, gt_count
