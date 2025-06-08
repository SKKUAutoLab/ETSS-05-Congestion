import torch
import shutil
import os

def save_checkpoint(state, mae_is_best, mse_is_best, path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(path, filename))
    epoch = state['epoch']
    if mae_is_best:
        shutil.copyfile(os.path.join(path, filename), os.path.join(path, 'epoch' + str(epoch) + '_best_mae.pth.tar'))
    if mse_is_best:
        shutil.copyfile(os.path.join(path, filename), os.path.join(path, 'epoch' + str(epoch) + '_best_mse.pth.tar'))

def crop_img_patches(img, size=512):
    w = img.shape[3]
    h = img.shape[2]
    x = int(w / size) + 1
    y = int(h / size) + 1
    crop_w = int(w / x)
    crop_h = int(h / y)
    patches = []
    for i in range(x):
        for j in range(y):
            start_x = crop_w * i
            if i == x - 1:
                end_x = w
            else:
                end_x = crop_w * (i + 1)
            start_y = crop_h * j
            if j == y - 1:
                end_y = h
            else:
                end_y = crop_h * (j + 1)
            sub_img = img[:, :, start_y:end_y, start_x:end_x]
            patches.append(sub_img)
    return patches