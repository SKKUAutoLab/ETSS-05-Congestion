#!/usr/bin/env python

# PyTorch port of the original TensorFlow 1.x implementation.
# The network architecture, loss, and active-learning loop are kept faithful to
# the original TF code base (see *_tf_backup.py); only the TF graph/session
# internals were replaced with eager PyTorch.

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms as tv_nms

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as image
from PIL import Image

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from skimage.feature import match_template

from aux_funcs import *
from count_gt_vs_output import *
from conf import get_image_info

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

global boxes_rej_to_user
global boxes_acc_to_user
global boxes_id_rej_to_user
global boxes_id_acc_to_user
global user_rej
global user_acc

global start_loc
global end_loc

global to_show
global fig2
global clicks_counter

clicks_counter = 0

boxes_rej_to_user = []
boxes_acc_to_user = []
boxes_id_rej_to_user = []
boxes_id_acc_to_user = []
user_rej = []
user_acc = []

training_iters = 1000
batch_size = 1

filt_mag = 0.1
bias_mag = 0.0


def _imresize(arr, factor):
    """
    Faithful replacement for the removed scipy.misc.imresize(arr, factor) with the
    defaults interp='bilinear', mode=None. Reproduces scipy's bytescale (min-max ->
    0..255 with clip + round-to-nearest, i.e. the +0.5 before the uint8 cast) followed
    by a PIL bilinear resize whose target size is truncated to int (as scipy does).
    """
    a = np.asarray(arr, dtype=np.float64)
    cmin, cmax = a.min(), a.max()
    cscale = cmax - cmin
    if cscale == 0:
        cscale = 1
    scale = 255.0 / cscale
    bytedata = (a - cmin) * scale
    b = (np.clip(bytedata, 0, 255) + 0.5).astype(np.uint8)
    h, w = b.shape[0], b.shape[1]
    newsize = (int(w * factor), int(h * factor))  # PIL size is (W, H); scipy truncates
    im = Image.fromarray(b)
    im = im.resize(newsize, Image.BILINEAR)
    return np.array(im)


class Network(nn.Module):
    """
    Forward pass. The network architecture is explained in the paper.
    Faithful port of the original TF `network` function:
      - crop the black frame -> conv(11x11, 3->nf1) + relu
      - conv(7x7, nf1->nf2) -> maxpool(2x2, keep argmax) + b2
      - per-location (across channels) L2 feature normalization (NaN->0) -> relu
        (this normalized map, pre-relu, is returned as `features`)
      - maxunpool(2x2) using the argmax locations
      - convT(5x5, nf2->nf1) + relu
      - conv(5x5, nf1->1)   (no trainable bias -- see note below)

    Note on the final-layer bias: the original code created it as
    tf.get_variable("eecoder_b4", ...) (a typo). The optimizer var-list and the
    L2 term both filter variable names on 'encoder'/'decoder', so this bias never
    matched, was never trained, and stayed 0 (bias_mag=0). We therefore use
    bias=False on conv4, which both reproduces the output exactly and makes the
    trainable-parameter set identical to the TF one.
    """

    def __init__(self, nf1, nf2, patch_sz_hf, len_x, len_y):
        super().__init__()
        self.hf = int(patch_sz_hf)
        self.len_x = int(len_x)
        self.len_y = int(len_y)
        self.nf2 = int(nf2)

        fil_sz_1, fil_sz_2, fil_sz_3, fil_sz_4 = 11, 7, 5, 5
        self.conv1 = nn.Conv2d(3, nf1, fil_sz_1, stride=1, padding=fil_sz_1 // 2)
        self.conv2 = nn.Conv2d(nf1, nf2, fil_sz_2, stride=1, padding=fil_sz_2 // 2)
        self.deconv3 = nn.ConvTranspose2d(nf2, nf1, fil_sz_3, stride=1, padding=fil_sz_3 // 2)
        self.conv4 = nn.Conv2d(nf1, 1, fil_sz_4, stride=1, padding=fil_sz_4 // 2, bias=False)
        self._init_weights()

    def _init_weights(self):
        # weights ~ N(0,1) * filt_mag ; biases = bias_mag (0)
        with torch.no_grad():
            for m in (self.conv1, self.conv2, self.deconv3, self.conv4):
                m.weight.normal_(0.0, 1.0).mul_(filt_mag)
                if m.bias is not None:
                    m.bias.fill_(bias_mag)

    def forward(self, x):
        hf, lx, ly, nf2 = self.hf, self.len_x, self.len_y, self.nf2
        # crop the black frame -> (1, 3, len_x, len_y)
        x = x[:, :, hf:hf + lx, hf:hf + ly]
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        # max-pool keeping argmax locations. The conv bias is per-channel, so
        # "pool then add bias" (TF) == "add bias then pool" (here); argmax unchanged.
        pooled, loc1 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        # per-location (across channels) L2 normalization, NaN -> 0
        norm = pooled.pow(2).sum(dim=1, keepdim=True).sqrt()
        features = pooled / norm
        features = torch.where(torch.isnan(features), torch.zeros_like(features), features)
        x = F.relu(features)
        x = F.max_unpool2d(x, loc1, kernel_size=2, stride=2, output_size=(lx, ly))
        x = F.relu(self.deconv3(x))
        G = self.conv4(x)
        # features for the loss: row-major (len_x/2, len_y/2) order to match the
        # numpy index mapping used elsewhere (calc_smaller_ind).
        feat_flat = features.permute(0, 2, 3, 1).reshape(-1, nf2)
        return G, feat_flat


def get_patches(all_scores_t, all_boxes_t, max_num_cells):
    """
    Getting the potential locations of the repeating object (NMS over the score map).
    Returns numpy arrays: selected box indices (sorted by score, truncated to
    max_num_cells), the positions of the positive-scored ones, the full score
    vector, and the selected scores.
    """
    keep = tv_nms(all_boxes_t, all_scores_t, 0.2)
    keep = keep[:max_num_cells]
    selected_scores = all_scores_t[keep]
    pos_ind = torch.nonzero(selected_scores > 0).reshape(-1)
    return [keep.detach().cpu().numpy(),
            pos_ind.detach().cpu().numpy(),
            all_scores_t.detach().cpu().numpy(),
            selected_scores.detach().cpu().numpy()]


def cost_G_obj(all_scores_t, features_t, rej_final, acc_final, rej_final_smaller, acc_final_smaller, nf2):
    """
    Calculate the loss (L_label + L_sub).
    rej_final / acc_final          : (len_x*len_y,)       numpy binary maps
    rej_final_smaller / acc_..._sm : (len_x/2*len_y/2,)   numpy binary maps
    """
    rej_idx = np.where(rej_final != 0)[0]
    acc_idx = np.where(acc_final == 1)[0]
    rej_idx_t = torch.as_tensor(rej_idx, dtype=torch.long, device=all_scores_t.device)
    acc_idx_t = torch.as_tensor(acc_idx, dtype=torch.long, device=all_scores_t.device)
    w_rej_final = all_scores_t[rej_idx_t]
    w_acc_final = all_scores_t[acc_idx_t]

    w_mult_v_pos = torch.mean((w_acc_final - 2) ** 2)
    w_mult_v_neg = torch.mean((w_rej_final + 2) ** 2)

    rej_idx_s = np.where(rej_final_smaller != 0)[0]
    acc_idx_s = np.where(acc_final_smaller == 1)[0]
    rej_idx_s_t = torch.as_tensor(rej_idx_s, dtype=torch.long, device=features_t.device)
    acc_idx_s_t = torch.as_tensor(acc_idx_s, dtype=torch.long, device=features_t.device)
    rej_features = features_t[rej_idx_s_t]
    acc_features = features_t[acc_idx_s_t]

    n = nf2 // 2
    cost = (w_mult_v_pos + w_mult_v_neg) \
        + torch.mean(rej_features[:, 0:n] ** 2) \
        + torch.mean(acc_features[:, n:] ** 2)
    return cost


def calc_cost_AE_prep(x_corr_np, th, all_boxes_t, len_x, len_y):
    """
    Calculate the initial positive and negative buckets using normalized cross
    correlation. x_corr_np: (1, len_x+2hf, len_y+2hf, 1) numpy.
    Returns (ind_pos, ind_neg): flat indices (into len_x*len_y) for the
    positive / negative initialization buckets.
    """
    scores_x_corr = np.reshape(
        x_corr_np[0, patch_sz_hf:patch_sz_hf + len_x, patch_sz_hf:patch_sz_hf + len_y, 0], [-1])
    ind = np.asarray(list(range((len_x) * (len_y))))
    scores_t = torch.as_tensor(scores_x_corr, dtype=torch.float32, device=all_boxes_t.device)

    selected_indices_pos = tv_nms(all_boxes_t, scores_t, 0.2).detach().cpu().numpy()
    selected_x_scores_pos = scores_x_corr[selected_indices_pos]
    selected_ind_pos = ind[selected_indices_pos]
    pos = np.where(selected_x_scores_pos > th)[0]
    ind_pos = np.squeeze(selected_ind_pos[pos])

    selected_indices_neg = tv_nms(all_boxes_t, torch.abs(scores_t), 0.2).detach().cpu().numpy()
    selected_x_scores_neg = scores_x_corr[selected_indices_neg]
    selected_ind_neg = ind[selected_indices_neg]
    neg = np.where(selected_x_scores_neg < 0)[0]
    ind_neg = np.squeeze(selected_ind_neg[neg])

    return [ind_pos, ind_neg]


def train_step(net, optimizer, x_orig_t, rej_final, acc_final, rej_smaller, acc_smaller, reg, nf2):
    """
    One optimizer step on (L_label + L_sub + reg * L2). L2 is summed over all
    trainable parameters (== the TF 'encoder'/'decoder' var set) as 0.5*sum(v^2).
    """
    optimizer.zero_grad()
    G, features = net(x_orig_t)
    all_scores_t = G.reshape(-1)
    cost = cost_G_obj(all_scores_t, features, rej_final, acc_final, rej_smaller, acc_smaller, nf2)
    lossl2 = 0.0
    for p in net.parameters():
        lossl2 = lossl2 + 0.5 * (p ** 2).sum()
    loss = cost + reg * lossl2
    loss.backward()
    optimizer.step()
    return loss


def show_to_user(input, boxes_rej, boxes_acc, acc_map, center_rej, center_acc, step):
    """
    Present the negative and the positive queries to the user
    """

    global to_show
    global plt_to_user
    global fig2
    global IS_EXP_MODE
    global to_show_offline

    dims = np.shape(input)
    to_show = np.zeros((dims[1], dims[2], 3))
    to_show = input[0, :, :, :].copy()
    for i in range(0, len(boxes_rej)):
        to_show[boxes_rej[i][0]:boxes_rej[i][2], boxes_rej[i][1], :] = [1, 0, 0]  # [0.33, 0, 0]
        to_show[boxes_rej[i][0]:boxes_rej[i][2], boxes_rej[i][1] + 1, :] = [1, 0, 0]  # [0.33, 0, 0]
        to_show[boxes_rej[i][0]:boxes_rej[i][2], boxes_rej[i][3] - 1, :] = [1, 0, 0]  # [0.33, 0, 0]

        to_show[boxes_rej[i][0], boxes_rej[i][1]:boxes_rej[i][3], :] = [1, 0, 0]  # [0.33, 0, 0]
        to_show[boxes_rej[i][0] + 1, boxes_rej[i][1]:boxes_rej[i][3], :] = [1, 0, 0]  # [0.33, 0, 0]
        to_show[boxes_rej[i][2], boxes_rej[i][1]:boxes_rej[i][3], :] = [1, 0, 0]  # [0.33, 0, 0]
        to_show[boxes_rej[i][2] - 1, boxes_rej[i][1]:boxes_rej[i][3], :] = [1, 0, 0]  # [0.33, 0, 0]

        to_show[center_rej[i][0], center_rej[i][1], :] = [0, 0, 0]  # [0.33, 0, 0]

    for i in range(0, len(boxes_acc)):
        to_show[boxes_acc[i][0]:boxes_acc[i][2], boxes_acc[i][1], :] = [0, 1, 0]  # [0, 0.33, 0]
        to_show[boxes_acc[i][0]:boxes_acc[i][2], boxes_acc[i][1] + 1, :] = [0, 1, 0]  # [0, 0.33, 0]
        to_show[boxes_acc[i][0]:boxes_acc[i][2], boxes_acc[i][3], :] = [0, 1, 0]  # [0, 0.33, 0]
        to_show[boxes_acc[i][0]:boxes_acc[i][2], boxes_acc[i][3] - 1, :] = [0, 1, 0]  # [0, 0.33, 0]

        to_show[boxes_acc[i][0], boxes_acc[i][1]:boxes_acc[i][3], :] = [0, 1, 0]  # [0, 0.33, 0]
        to_show[boxes_acc[i][0] + 1, boxes_acc[i][1]:boxes_acc[i][3], :] = [0, 1, 0]  # [0, 0.33, 0]
        to_show[boxes_acc[i][2], boxes_acc[i][1]:boxes_acc[i][3], :] = [0, 1, 0]  # [0, 0.33, 0]
        to_show[boxes_acc[i][2] - 1, boxes_acc[i][1]:boxes_acc[i][3], :] = [0, 1, 0]  # [0, 0.33, 0]

        to_show[center_acc[i][0], center_acc[i][1], :] = [0, 0, 0]  # [0.33, 0, 0]

    plt.close(1)
    fig2 = plt.figure(2, figsize=(80, 60))
    to_show_offline = to_show.copy()
    fig2.canvas.mpl_connect('button_press_event', onclick_userCorrection)

    plt_to_user = plt.imshow(to_show)
    plt.show()


def onclick_userCorrection(event):
    """
    Event when the user clicks on the image. In case it is inside a negative or a positive window, this window
    is changed to have positive ot negative label respectively.
    In this version we removed the option for 'unchanged' button. It can be integrated easily.
    """

    global boxes_rej_to_user
    global boxes_acc_to_user
    global boxes_id_rej_to_user
    global boxes_id_acc_to_user
    global user_rej
    global user_acc
    global to_show
    global plt_to_user
    global fig2
    global clicks_counter
    global step

    [x, y] = event.xdata, event.ydata
    for i in range(0, len(boxes_rej_to_user)):
        # if the click is indie a negative window - mark it in black and add it to the positive bucket labels
        if y > boxes_rej_to_user[i][0] and y < boxes_rej_to_user[i][2] and x > boxes_rej_to_user[i][1] and x < \
                boxes_rej_to_user[i][3]:
            user_acc.append(boxes_id_rej_to_user[i])
            to_show[boxes_rej_to_user[i][0]:boxes_rej_to_user[i][2], boxes_rej_to_user[i][1], :] = [0, 0, 0]
            to_show[boxes_rej_to_user[i][0]:boxes_rej_to_user[i][2], boxes_rej_to_user[i][3], :] = [0, 0, 0]
            to_show[boxes_rej_to_user[i][0], boxes_rej_to_user[i][1]:boxes_rej_to_user[i][3], :] = [0, 0, 0]
            to_show[boxes_rej_to_user[i][2], boxes_rej_to_user[i][1]:boxes_rej_to_user[i][3], :] = [0, 0, 0]
            plt_to_user.set_data(to_show)
            clicks_counter += 1
            break

    for i in range(0, len(boxes_acc_to_user)):
        # if the click is indie a positive window - mark it in black and add it to the negative bucket labels
        if y > boxes_acc_to_user[i][0] and y < boxes_acc_to_user[i][2] and x > boxes_acc_to_user[i][1] and x < \
                boxes_acc_to_user[i][3]:
            user_rej.append(boxes_id_acc_to_user[i])
            to_show[boxes_acc_to_user[i][0]:boxes_acc_to_user[i][2], boxes_acc_to_user[i][1], :] = [0, 0, 0]
            to_show[boxes_acc_to_user[i][0]:boxes_acc_to_user[i][2], boxes_acc_to_user[i][3], :] = [0, 0, 0]
            to_show[boxes_acc_to_user[i][0], boxes_acc_to_user[i][1]:boxes_acc_to_user[i][3], :] = [0, 0, 0]
            to_show[boxes_acc_to_user[i][2], boxes_acc_to_user[i][1]:boxes_acc_to_user[i][3], :] = [0, 0, 0]
            plt_to_user.set_data(to_show)
            clicks_counter += 1
            break
    fig2.canvas.draw()
    fig2.canvas.flush_events()


def should_stop(rej_final, acc_final, all_scores):
    """
    Stopping condition. If all the locations of the negative labels received a low
    value and the locations of the positive labels high value (in the network output).
    """

    if np.sum(all_scores[np.where(rej_final != 0)] >= -0.4) != 0:
        return False

    if np.sum(all_scores[np.where(acc_final == 1)] <= 0.9) != 0:
        return False

    return True


def purple_dots_on_image(input, ind_to_ask, acc_ind_final):
    """
    Create the method's solution image. This method outputs the input image with the locations of the repeating objects
    """
    global all_center_ind
    final_res_image = np.zeros((np.shape(input)[1], np.shape(input)[2], 3))
    gt_color_1 = [0.64, 0.27, 0.61]

    for i in range(0, 3):
        final_res_image[:, :, i] = (input[0, :, :, 0] + input[0, :, :, 1] + input[0, :, :, 2]) / 3
    for i in range(0, len(ind_to_ask)):
        final_res_image[all_center_ind[ind_to_ask[i]][0] - 2:all_center_ind[ind_to_ask[i]][0] + 2, all_center_ind[ind_to_ask[i]][1] - 2:all_center_ind[ind_to_ask[i]][1] + 2, 0] = gt_color_1[0]
        final_res_image[all_center_ind[ind_to_ask[i]][0] - 2:all_center_ind[ind_to_ask[i]][0] + 2, all_center_ind[ind_to_ask[i]][1] - 2:all_center_ind[ind_to_ask[i]][1] + 2, 1] = gt_color_1[1]
        final_res_image[all_center_ind[ind_to_ask[i]][0] - 2:all_center_ind[ind_to_ask[i]][0] + 2, all_center_ind[ind_to_ask[i]][1] - 2:all_center_ind[ind_to_ask[i]][1] + 2, 2] = gt_color_1[2]
    acc_final = np.where(acc_ind_final == 1)[0]
    for i in range(0, len(acc_final)):
        final_res_image[all_center_ind[acc_final[i]][0] - 2:all_center_ind[acc_final[i]][0] + 2, all_center_ind[acc_final[i]][1] - 2:all_center_ind[acc_final[i]][1] + 2, 0] = gt_color_1[0]
        final_res_image[all_center_ind[acc_final[i]][0] - 2:all_center_ind[acc_final[i]][0] + 2, all_center_ind[acc_final[i]][1] - 2:all_center_ind[acc_final[i]][1] + 2, 1] = gt_color_1[1]
        final_res_image[all_center_ind[acc_final[i]][0] - 2:all_center_ind[acc_final[i]][0] + 2, all_center_ind[acc_final[i]][1] - 2:all_center_ind[acc_final[i]][1] + 2, 2] = gt_color_1[2]

    final_res_image = final_res_image[patch_sz_hf:-patch_sz_hf, patch_sz_hf:-patch_sz_hf, :]
    count = len(acc_final) + len(np.setdiff1d(ind_to_ask, acc_final))

    return [count, final_res_image]


def adding_rej(path, len_x, len_y, rej_ind_final):
    """
    Adding negative points in borders
    """
    tmp_rej = np.reshape(rej_ind_final, (len_x, len_y))
    tmp_rej[0:2, :] = 1
    tmp_rej[:, 0:2] = 1
    tmp_rej[-2:, :] = 1
    tmp_rej[:, -2:] = 1
    rej_ind_final = np.reshape(tmp_rej, [-1])
    return rej_ind_final


def calc_smaller_ind(curr_pos_ind, len_x, len_y, factor, output_mat=False):
    """
    Mapping between the image presnted to the user (which is bigger by a factor of 2) and the
    real size of the image.
    """
    mat = np.zeros((len_x, len_y))
    mat_smaller = np.zeros((np.int32(len_x / factor), np.int32(len_y / factor)))

    [row, col] = np.unravel_index(curr_pos_ind, [len_x, len_y])
    row_small = np.int32(row / factor)
    col_small = np.int32(col / factor)
    for i in range(0, len(row_small)):
        mat_smaller[row_small[i], col_small[i]] = 1
    if output_mat:
        return np.reshape(mat_smaller, -1)
    else:
        return np.where(np.reshape(mat_smaller == 1, -1))[0]


def calc_larger_ind(curr_pos_ind, len_x, len_y, factor):
    """
    Mapping between the input image and the image presnted to the user (which is bigger
    by a factor of 2).
    """
    mat = np.zeros((len_x, len_y))
    mat_larger = np.zeros((len_x * factor, len_y * factor))

    [row, col] = np.unravel_index(curr_pos_ind, [len_x, len_y])
    row_large = row * factor
    col_large = col * factor
    for i in range(0, len(row_large)):
        mat_larger[row_large[i], col_large[i]] = 1
    return np.where(np.reshape(mat_larger == 1, -1))[0]


def change_to_closest_to_ask(curr_ind, ind_to_ask, len_x, len_y):
    """
    Change curr_ind to the closest index from ind_to_ask
    """
    [row_curr, col_curr] = np.unravel_index(curr_ind, [len_x, len_y])
    [row_ind_to_ask, col_ind_to_ask] = np.unravel_index(ind_to_ask, [len_x, len_y])
    new_row = []
    new_col = []
    for i in range(0, len(row_curr)):
        d = []
        for j in range(0, len(row_ind_to_ask)):
            d.append(np.power(row_curr[i] - row_ind_to_ask[j], 2) + np.power(col_curr[i] - col_ind_to_ask[j], 2))
        new_row.append(row_ind_to_ask[np.argmin(d)])
        new_col.append(col_ind_to_ask[np.argmin(d)])

    mat = np.zeros((len_x, len_y))
    for i in range(0, len(new_row)):
        mat[new_row[i], new_col[i]] = 1
    return np.where(np.reshape(mat, -1) == 1)[0]


def main(dir, base_dir, im_size, image_name, window_loc, number_of_patches, th, filename, gt_color, show_gray, max_num_cells, participant_name):
    init_time = time.time()
    global boxes_rej_to_user
    global boxes_acc_to_user
    global boxes_id_rej_to_user
    global boxes_id_acc_to_user
    global user_rej
    global user_acc
    global step

    global start_loc
    global end_loc
    global all_boxes
    global all_center_ind
    global nf1
    global nf2
    global patch_sz_hf

    nf1 = number_of_patches
    nf2 = 2 * nf1

    factor = 2

    len_x = im_size[0]
    len_y = im_size[1]

    y = np.int32(np.linspace(0, len_x - 1, len_x))
    x = np.int32(np.linspace(0, len_y - 1, len_y))
    [i, j] = np.meshgrid(x, y)
    all_boxes = np.zeros(((len_x) * (len_y), 4), np.int32)
    all_boxes[:, 0] = np.reshape(j, -1)
    all_boxes[:, 1] = np.reshape(i, -1)
    all_boxes[:, 2] = np.reshape(j, -1) + patch_sz - 1
    all_boxes[:, 3] = np.reshape(i, -1) + patch_sz - 1

    all_center_ind = np.zeros(((len_x) * (len_y), 2), np.int32)
    all_center_ind[:, 0] = np.reshape(j, -1) + patch_sz_hf
    all_center_ind[:, 1] = np.reshape(i, -1) + patch_sz_hf

    all_boxes_t = torch.as_tensor(all_boxes, dtype=torch.float32, device=device)

    output_dir = dir + "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    learning_rate_G = 0.001
    learning_rate_G_init = 0.001

    # network + the two optimizers (init vs. main), mirroring the TF AdamOptimizers
    net = Network(nf1, nf2, patch_sz_hf, len_x, len_y).to(device)
    opt_G_rejacc_maps_init = torch.optim.Adam(net.parameters(), lr=learning_rate_G_init)
    opt_G_rejacc_maps = torch.optim.Adam(net.parameters(), lr=learning_rate_G)

    num = (len_x) * (len_y)
    num_smaller = int((len_x / factor) * (len_y / factor))

    # binary maps over the (vectorized) input image
    rej_ind_final_from_AE = np.zeros((num,))
    acc_ind_final_from_AE = np.zeros((num,))
    rej_ind_final = np.zeros((num,))
    acc_ind_final = np.zeros((num,))
    rej_ind_final_smaller = np.zeros((num_smaller,))
    acc_ind_final_smaller = np.zeros((num_smaller,))
    curr_ind_box = []

    batch_x_orig = read_dataset(filename, len_x, len_y, patch_sz, show_gray)
    # the network input, NCHW and constant (not trained)
    x_orig_t = torch.as_tensor(batch_x_orig, dtype=torch.float32, device=device).permute(0, 3, 1, 2).contiguous()

    # there can be more than one window location
    start_loc = window_loc

    dims = batch_x_orig.shape
    img_corr = np.zeros((dims[1], dims[2], len(start_loc)))
    for i in range(0, len(start_loc)):
        patch = batch_x_orig[0, start_loc[i][1]:start_loc[i][1] + patch_sz, start_loc[i][0]:start_loc[i][0] + patch_sz, :]
        res = np.zeros((dims[1] - patch_sz_hf * 2, dims[2] - patch_sz_hf * 2))
        # calculate the normalize cross correlation between the input image and the initialized repeating object window
        for ch in range(0, 3):
            res = res + match_template(batch_x_orig[0, :, :, ch], patch[:, :, ch])

        res = res / 3
        img_corr[patch_sz_hf:-patch_sz_hf, patch_sz_hf:-patch_sz_hf, i] = res

    plt.show()

    for i in range(0, len(start_loc)):
        batch_x_corr = np.zeros((1, len_x + patch_sz_hf * 2, len_y + patch_sz_hf * 2, 1))
        batch_x_corr[0, :, :, 0] = img_corr[:, :, i]

        # extract initialized positive and negative buckets from the normalized-correlation image
        [x_corr_pos_curr, x_corr_rej_curr] = calc_cost_AE_prep(batch_x_corr, th, all_boxes_t, len_x, len_y)
        rej_ind_final_from_AE[x_corr_rej_curr] = 1
        acc_ind_final_from_AE[x_corr_pos_curr] = 1

        rej_ind_final[x_corr_rej_curr] = 1
        acc_ind_final[x_corr_pos_curr] = 1
        # adding rej points in borders
        rej_ind_final = adding_rej(filename, len_x, len_y, rej_ind_final)

        rej_ind_final_smaller = calc_smaller_ind(np.where(rej_ind_final == 1)[0], len_x, len_y, factor, True)
        acc_ind_final_smaller = calc_smaller_ind(np.where(acc_ind_final == 1)[0], len_x, len_y, factor, True)

    regularization_G = 0
    # Initialize step: train the network with the initialize labels
    with torch.no_grad():
        G, _ = net(x_orig_t)
    curr_all_scores = G.reshape(-1).detach().cpu().numpy()
    i = 0
    while not should_stop(rej_ind_final, acc_ind_final, curr_all_scores):
        i += 1
        train_step(net, opt_G_rejacc_maps_init, x_orig_t, rej_ind_final, acc_ind_final,
                   rej_ind_final_smaller, acc_ind_final_smaller, regularization_G, nf2)
        with torch.no_grad():
            G, _ = net(x_orig_t)
        curr_all_scores = G.reshape(-1).detach().cpu().numpy()

    step = 0
    input = batch_x_orig
    G_th = 0

    regularization_G = 0.001

    while step * batch_size <= training_iters:
        with torch.no_grad():
            G, features = net(x_orig_t)
        curr_features = features.detach().cpu().numpy()
        all_scores_t = G.reshape(-1)

        [curr_ind_box, curr_pos_ind, curr_all_scores, _] = get_patches(all_scores_t, all_boxes_t, max_num_cells)
        curr_center_ind = all_center_ind[curr_ind_box]
        curr_ind_box = curr_ind_box[0:curr_pos_ind.shape[0]]
        curr_ind_box_small = calc_smaller_ind(curr_pos_ind, len_x, len_y, factor)

        potential_map = np.zeros((len_x) * (len_y))
        potential_map[curr_ind_box[0:curr_pos_ind.shape[0]]] = 1
        potential_map_tmp = np.where(potential_map == 1)[0]
        potential_map = np.reshape(potential_map, (len_x, len_y))
        ind_G = np.squeeze(np.where(potential_map == 1))

        acc_ind_final_new = np.reshape(acc_ind_final, (len_x, len_y))
        ind_final = np.squeeze(np.where(acc_ind_final_new == 1))
        if ind_final.ndim == 1:
            ind_final = np.expand_dims(ind_final, 1)
        if ind_G.ndim == 1:
            ind_G = np.expand_dims(ind_G, 1)

        # If the new candidiates fall very close to the ones there were allready labeled - ignore them
        ignore_from_show = []
        for i in range(0, ind_G.shape[1]):
            for j in range(0, ind_final.shape[1]):
                if (ind_G[0][i] > ind_final[0][j] - patch_sz / 3) and (ind_G[0][i] < ind_final[0][j] + patch_sz / 3) and (
                        ind_G[1][i] > ind_final[1][j] - patch_sz / 3) and (ind_G[1][i] < ind_final[1][j] + patch_sz / 3) and (
                        ind_final[0][j] != ind_G[0][i] or ind_final[1][j] != ind_G[1][i]):
                    acc_ind_final_new[ind_final[0][j]][ind_final[1][j]] = 1
                    acc_ind_final_new[ind_G[0][i]][ind_G[1][i]] = 0
                    ignore_from_show.append(potential_map_tmp[i])
                    acc_ind_final = np.reshape(acc_ind_final_new, [-1])

        curr_features_acc = curr_features[calc_smaller_ind(np.where(acc_ind_final == 1), len_x, len_y, factor)]
        curr_features_rej = curr_features[calc_smaller_ind(np.where(rej_ind_final == 1), len_x, len_y, factor)]
        # Concatenate the features that are related to positive and negative labeled windows
        tot_features = np.concatenate((curr_features_acc, curr_features_rej))
        tot_features_bool = np.concatenate((np.ones((curr_features_acc.shape[0])), -np.ones((curr_features_rej.shape[0]))))

        # Calculate nearest neigbour in feature space
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(tot_features)

        ind_to_ask = curr_ind_box
        ind_to_ask = np.setdiff1d(ind_to_ask, ignore_from_show)
        # remove index that are already labeled
        ind_to_ask = np.setdiff1d(ind_to_ask, np.where((rej_ind_final + acc_ind_final) >= 1))
        ind_to_ask_small = calc_smaller_ind(ind_to_ask, len_x, len_y, factor)

        # calculate the distance for each potential location
        distances, indices = nbrs.kneighbors(curr_features[ind_to_ask_small])

        nn = tot_features_bool[indices]
        nn_pos = np.where(nn == 1)[0]
        nn_neg = np.where(nn == -1)[0]

        if "coke" in filename:
            number_to_show = 2
        else:
            number_to_show = 5

        max_clusters = number_to_show * 2

        # Sample user queries - with poisitive label
        a = []
        if len(ind_to_ask_small[nn_pos]) > 0:
            # cluster the potential locations that are realted to the positive windows
            n_clusters = np.min((max_clusters, len(ind_to_ask_small[nn_pos])))
            kmeans_pos = KMeans(n_clusters=n_clusters, random_state=0).fit(curr_features[ind_to_ask_small[nn_pos]])
            a_candidates = []
            farer_dist_pos = np.argsort(distances[nn_pos, 0])[::-1]
            pos_labels = np.unique(kmeans_pos.labels_)
            # for each cluster, pick the farthest candidate
            for i in range(0, len(pos_labels)):
                worst_dist_ind = np.where(kmeans_pos.labels_[farer_dist_pos] == pos_labels[i])[0][0]
                # distances has shape (N, 1); take the scalar distance (numpy >=1.24
                # rejects the ragged array the original relied on -- same value/intent).
                worst_dist = distances[nn_pos, 0][farer_dist_pos[worst_dist_ind]]
                a_candidates.append([worst_dist, worst_dist_ind])
            a_candidates = np.asarray(a_candidates)
            a = ind_to_ask_small[nn_pos][farer_dist_pos[np.int32(a_candidates[np.argsort(a_candidates[:, 0])][-number_to_show:][:, 1])]]

        # Sample user queries - with negative label
        r = []
        if len(ind_to_ask_small[nn_neg]) > 0:
            # cluster the potential locations that are realted to the negative windows
            n_clusters = np.min((max_clusters, len(ind_to_ask_small[nn_neg])))
            kmeans_neg = KMeans(n_clusters=n_clusters, random_state=0).fit(curr_features[ind_to_ask_small[nn_neg]])
            r_candidates = []
            farer_dist_neg = np.argsort(distances[nn_neg, 0])[::-1]
            neg_labels = np.unique(kmeans_neg.labels_)
            # for each cluster, pick the farthest candidate
            for i in range(0, len(neg_labels)):
                worst_dist_ind = np.where(kmeans_neg.labels_[farer_dist_neg] == neg_labels[i])[0][0]
                # distances has shape (N, 1); take the scalar distance (numpy >=1.24
                # rejects the ragged array the original relied on -- same value/intent).
                worst_dist = distances[nn_neg, 0][farer_dist_neg[worst_dist_ind]]
                r_candidates.append([worst_dist, worst_dist_ind])
            r_candidates = np.asarray(r_candidates)
            r = ind_to_ask_small[nn_neg][farer_dist_neg[np.int32(r_candidates[np.argsort(r_candidates[:, 0])][-number_to_show:][:, 1])]]

        if len(a) > 0:
            a = calc_larger_ind(a, np.int32(len_x / factor), np.int32(len_y / factor), factor)
            a = change_to_closest_to_ask(a, ind_to_ask, len_x, len_y)
        if len(r) > 0:
            r = calc_larger_ind(r, np.int32(len_x / factor), np.int32(len_y / factor), factor)
            r = change_to_closest_to_ask(r, ind_to_ask, len_x, len_y)
        # 'r' and 'a' are the locations to ask the user

        # if all the potential locations have high score in the network's output map or the algorithm reached 5 iterations - stop
        if len(np.where(curr_all_scores[ind_to_ask] < 0.85)[0]) == 0 or step > 4:  # stop the algorithm
            final(dir, image_name, step, input, ind_to_ask, acc_ind_final, init_time, clicks_counter, patch_sz)
            break

        boxes_id_rej_to_user = np.int32(np.setdiff1d(r, np.where((rej_ind_final + acc_ind_final) >= 1)))
        boxes_id_acc_to_user = np.int32(np.setdiff1d(a, np.where((acc_ind_final + rej_ind_final) >= 1)))

        # convert the queries locations to windows
        boxes_rej_to_user = all_boxes[boxes_id_rej_to_user]
        all_center_ind_rej = all_center_ind[boxes_id_rej_to_user]
        boxes_acc_to_user = all_boxes[boxes_id_acc_to_user]
        all_center_ind_acc = all_center_ind[boxes_id_acc_to_user]
        plt.close()
        plt.figure(1)

        # ask the user
        if (boxes_id_rej_to_user.shape[0] > 0 or boxes_id_acc_to_user.shape[0] > 0):
            show_to_user(input, boxes_rej_to_user, boxes_acc_to_user, acc_ind_final, all_center_ind_rej, all_center_ind_acc, step)

        user_tot = np.concatenate((user_rej, user_acc), axis=0)
        user_rej = np.int32(np.concatenate((user_rej, np.setdiff1d(boxes_id_rej_to_user, user_tot)), axis=0))
        user_acc = np.int32(np.concatenate((user_acc, np.setdiff1d(boxes_id_acc_to_user, user_tot)), axis=0))

        user_rej = np.setdiff1d(user_rej, -1)
        user_acc = np.setdiff1d(user_acc, -1)
        # update the final positive and negative labels according to the user decision
        rej_ind_final[user_rej] = 1
        acc_ind_final[user_acc] = 1

        if np.where(acc_ind_final == 1)[0].shape[0] != np.setdiff1d(np.where(acc_ind_final == 1)[0], np.where(rej_ind_final == 1)[0]).shape[0]:
            print("Should not get into thie function")
            import IPython; IPython.embed()

        rej_ind_final_smaller = calc_smaller_ind(np.where(rej_ind_final == 1)[0], len_x, len_y, factor, True)
        acc_ind_final_smaller = calc_smaller_ind(np.where(acc_ind_final == 1)[0], len_x, len_y, factor, True)

        user_rej = []
        user_acc = []
        cost_G = []
        # Train the network using the updated labels
        while not should_stop(rej_ind_final, acc_ind_final, curr_all_scores):
            train_step(net, opt_G_rejacc_maps, x_orig_t, rej_ind_final, acc_ind_final,
                       rej_ind_final_smaller, acc_ind_final_smaller, regularization_G, nf2)
            with torch.no_grad():
                G, _ = net(x_orig_t)
            curr_all_scores = G.reshape(-1).detach().cpu().numpy()

        step += 1


def final(dir, image_name, step, input, ind_to_ask, acc_ind_final, init_time, clicks_counter, patch_sz):
    """
    The end of the algorithm: (1) save the input image with the repeating object on it. (2) save the repeating object in a .txt file.
    (3) calculate the localization error (4) print total time, clicks, false positive/negative, counting, ground_truth
    (5) save these measurmnets in a .txt file
    """
    print("----------------------exit---------------")
    [count, final_res_image] = purple_dots_on_image(input, ind_to_ask, acc_ind_final)
    image.imsave(dir + 'final_res.png', _imresize(final_res_image, 2.0))

    np.savetxt(dir + 'res_ours_' + str(step) + '.txt', np.concatenate((np.setdiff1d(ind_to_ask, np.where(acc_ind_final == 1)), np.where(acc_ind_final == 1)[0])))
    print("COUNT: ", count)
    stop_time = time.time() - init_time
    print("TOTAL TIME", stop_time)
    print("Number of clicks: ", clicks_counter)
    print("----------------------FP and FN---------------")
    [FP, FN, gt, count_ours] = count_images(image_name, dir, step, patch_sz)

    print("gt_count: ", gt[0])
    print("our count: ", count_ours[0])
    print("FN: ", FN[0])
    print("FP: ", FP[0])

    print(str(np.round(stop_time, 4)))

    output_file = open(dir + "/data.txt", "a+")
    output_file.write("-----------------------------\n")
    output_file.write("Step: " + str(step) + "\n")
    output_file.write("TOTAL TIME: " + str(np.round(stop_time, 4)) + "\n")
    output_file.write("Clicks : " + str(clicks_counter) + "\n")
    output_file.write("Count Ours : " + str(count_ours[0]) + "\n")
    output_file.write("gt : " + str(gt[0]) + "\n")
    output_file.write("FP : " + str(FP[0]) + "\n")
    output_file.write("FN : " + str(FN[0]) + "\n")
    output_file.write("-----------------------------\n")
    output_file.close()

    plt.figure(10)
    plt.imshow(final_res_image, cmap='gray')
    plt.show()


image_name = sys.argv[1]
participant_name = sys.argv[2]
print("image_name:", image_name)
print("participant_name:", participant_name)

[im_size, window_loc, number_of_patches, th, path, show_gray, max_num_cells, gt_color, curr_patch_sz] = get_image_info(image_name)
patch_sz = curr_patch_sz
patch_sz_hf = np.int32(np.floor(curr_patch_sz / 2))

dir = "user_study/" + str(participant_name) + "/" + str(image_name) + "/"
if not os.path.exists(dir):
    os.makedirs(dir)
print(dir)

main(dir, dir, im_size, image_name, window_loc, number_of_patches, th, path, gt_color, show_gray, max_num_cells, participant_name)
