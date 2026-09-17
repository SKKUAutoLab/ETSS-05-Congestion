# _*_ coding: utf-8 _*_
# @author   : wfs
# @time     : 2022/1/18 17:24
# @File     : vis_grad_cam.py
# @Software : PyCharm

import cv2
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from Networks import JCTNet

class ShowGradCam:
    def __init__(self, conv_layer):
        assert isinstance(conv_layer, torch.nn.Module), "input layer should be torch.nn.Module"
        self.conv_layer = conv_layer
        self.conv_layer.register_forward_hook(self.forward_hook)
        self.conv_layer.register_backward_hook(self.backward_hook)
        self.grad_res = []
        self.feature_res = []

    def backward_hook(self, module, grad_in, grad_out):
        self.grad_res.append(grad_out[0].detach())

    def forward_hook(self, module, input, output):
        self.feature_res.append(output)

    def gen_cam(self, feature_map, grads):
        """
        依据梯度和特征图，生成cam
        :param feature_map: np.array， in [C, H, W]
        :param grads: np.array， in [C, H, W]
        :return: np.array, [H, W]
        """
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)
        weights = np.mean(grads, axis=(1, 2))  #

        for i, w in enumerate(weights):
            cam += w * feature_map[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (32, 32))
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam

    def show_on_img(self, input_img):
        '''
        write heatmap on target img with color bar
        :param input_img: cv2:ndarray/img_pth
        :return: save jpg
        '''
        if isinstance(input_img, str):
            input_img = cv2.imread(input_img)
        
        img_size = (input_img.shape[1], input_img.shape[0])
        fmap = self.feature_res[0].cpu().data.numpy().squeeze()
        grads_val = self.grad_res[0].cpu().data.numpy().squeeze()
        
        cam = self.gen_cam(fmap, grads_val)
        cam = cv2.resize(cam, img_size)
        
        # 生成热力图并归一化到0-255
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET) / 255.0
        
        # 叠加原始图像
        cam = heatmap + np.float32(input_img/255.)
        cam = cam / np.max(cam) * 255
        
        result = add_colorbar_legend(cam, cv2.COLORMAP_JET, 0.0, 1.0)
        # 保存结果
        cv2.imwrite('./vis/heatmap.png', result, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        print('save gradcam result with color bar in grad_feature.png')

def img_transform(img_in, transform):
     """
     将img进行预处理，并转换成模型输入所需的形式—— B*C*H*W
     :param img_roi: np.array
     :return:
     """
     img = img_in.copy()
     img = Image.fromarray(np.uint8(img))
     img = transform(img)
     img = img.unsqueeze(0)    # C*H*W --> B*C*H*W
     return img
 
def img_preprocess(img_in):
     """
     读取图片，转为模型可读的形式
     :param img_in: ndarray, [H, W, C]
     :return: PIL.image
     """
     img = img_in.copy()
     img = img[:, :, ::-1]   # BGR --> RGB
     H, W = img.shape[:2]
     H, W = H // 32 * 32, W // 32 * 32
     img = cv2.resize(img, (W, H))
     transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
     ])
     img_input = img_transform(img, transform)
     return img_input        


def add_colorbar_legend(image, colormap=cv2.COLORMAP_JET, min_val=0.0, max_val=1.0):
    """在图像右下角添加颜色条和数值范围"""
    h, w = image.shape[:2]

    # 创建颜色条
    colorbar_width = 50
    colorbar_height = image.shape[0]
    margin = 0  # 边距

    # 生成颜色条渐变
    colorbar = np.linspace(0, 255, colorbar_height).astype(np.uint8)
    colorbar = np.repeat(colorbar.reshape(-1, 1), colorbar_width, axis=1)
    colorbar = cv2.applyColorMap(colorbar, colormap)

    # 放置颜色条位置
    start_x = w - colorbar_width - margin
    start_y = h - colorbar_height - margin

    # 确保不超出图像范围
    if start_x < 0 or start_y < 0:
        print("图像太小，无法添加颜色条")
        return image

    # 将颜色条叠加到图像上
    image[start_y:start_y+colorbar_height, start_x:start_x+colorbar_width] = colorbar

    # 添加数值文本
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 255, 255)  # 白色文本

    # 最大值文本
    cv2.putText(image, f"{max_val:.1f}", 
               (start_x + colorbar_width + 5, start_y + 20), 
               font, font_scale, color, thickness, cv2.LINE_AA)

    # 最小值文本
    cv2.putText(image, f"{min_val:.1f}", 
               (start_x + colorbar_width + 5, start_y + colorbar_height - 10), 
               font, font_scale, color, thickness, cv2.LINE_AA)

    return image


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # set vis gpu
    device = torch.device('cuda')

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path_img = os.path.join("data/ShanghaiTech/part_A_final/test_data/images/IMG_11.jpg")
    model_path = os.path.join('ckpts/model_sh_A.pth')

    # 图片读取；网络加载
    img = cv2.imread(path_img, 1)  # H*W*C
    img_input = img_preprocess(img).to(device)
    model = JCTNet.Net()
    model.to(device)
    checkpoint = torch.load(model_path, device)
    model.load_state_dict(checkpoint)
    model.eval()

    gradCam = ShowGradCam(model.conv_before_upsample) #............................. def which layer to show

    # forward
    output = model(img_input)
    print(output.size(), torch.sum(output))

    # backward
    model.zero_grad()
    class_loss = torch.sum(output)
    class_loss.backward()

    # save result
    gradCam.show_on_img(img) #.......................... show gradcam on target pic
