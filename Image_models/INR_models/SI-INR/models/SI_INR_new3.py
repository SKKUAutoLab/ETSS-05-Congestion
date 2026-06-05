import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from models.FourierEncoding import PositionalEncoding
from models.ses_cov import SESConv_H_H, SESConv_Z2_H, SESConv_H_H_1x1, SESMaxProjection
from models.pys_model import PyConvHead
from models.GCT import GCT

def get_coordinates(batch_size, res_original_w, res_original_h, k_HR, device, lower = -0.99, upper = 0.99, endpoint_s = True):
    x = np.linspace(lower, upper, res_original_h, endpoint = endpoint_s)
    y = np.linspace(lower, upper, res_original_w, endpoint = endpoint_s)
    xx, yy = np.meshgrid(x, y)
    dx_HR, dy_HR = [], []
    for i in range(0, res_original_w, k_HR):
        tmp_x, tmp_y = [], []
        for j in range(0, res_original_h, k_HR):
            tmp_x.append(xx[i][j])
            tmp_y.append(yy[i][j])
        dx_HR.append(tmp_x)
        dy_HR.append(tmp_y)
    d_HR = np.array([dx_HR, dy_HR])
    del dx_HR, dy_HR, xx, yy
    cor_map= torch.tensor(np.transpose(np.reshape(d_HR, (d_HR.shape[0], d_HR.shape[1]*d_HR.shape[2])), [1,0]), dtype=torch.float, device = device)
    cor_maps = cor_map.unsqueeze(0).repeat(batch_size, 1, 1) # [4, 1024, 2]
    return cor_maps

class SE_Encoder(nn.Module):
    def __init__(self,BatchNorm=nn.BatchNorm2d,k_size=3):
        super(SE_Encoder, self).__init__()
        self.features = VGG_Backbone()
        self.pyconvhead = PyConvHead(512, 512, BatchNorm)
        self.GCT = GCT(512, k_size)

    def forward(self, x): # [4, 3, 256, 256]
        x = self.features(x)
        x = F.interpolate(x, scale_factor=2)
        x_pyconv = self.pyconvhead(x)
        x_GCT = self.GCT(x)
        x = x_pyconv * x_GCT # [4, 512, 32, 32]
        return x

class New_bay_Net(nn.Module):
    def __init__(self):
        super(New_bay_Net, self).__init__()
        self.modelA = SE_Encoder()
        self.m = 64
        self.matrix_size = 256
        self.pos_encode_layer = PositionalEncoding(self.m)
        self.Encoder2z = Encoder2z()
        self.cc_decoder = build_cc_decoder(self.matrix_size)
        self.kl_div = 0

    def forward(self, x): # [4, 3, 256, 256]
        x = self.modelA(x)
        x = self.Encoder2z(x)
        b = x.shape[0]
        density_w = x.shape[2]
        density_h = x.shape[3]
        grid_new = get_coordinates(b, density_w, density_h, 1, x.device)
        grid_c1 = self.pos_encode_layer(grid_new)
        x, kl_div = self.cc_decoder(x, grid_c1)
        self.kl_div = kl_div
        return x # [4, 1, 64, 64]
    
class Encoder2z(nn.Module):
    def __init__(self):
        super(Encoder2z, self).__init__()
        self.frontend_feat5 = []
        self.frontend_feat5 += [256, 256]
        self.frontend5 = make_layers_4(self.frontend_feat5, in_channels = 512, batch_norm = True)
        scales = [0.8 * 1.11**i for i in range(3)]
        self.sesn1 = nn.Sequential(SESConv_Z2_H(256, 256, 3, 3, scales=scales, padding=1, bias=True, basis_type='A'),
                                   SESMaxProjection(), nn.LeakyReLU(True), nn.BatchNorm2d(256), nn.Conv2d(256, 256, kernel_size=1))
        self._initialize_weights() 

    def forward(self, feature_high): # [4, 512, 32, 32]
        feature_high = self.frontend5(feature_high)
        feature_high = self.sesn1(feature_high) + feature_high # [4, 256, 32, 32]
        return feature_high
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0.001)
            elif isinstance(m, (SESConv_H_H, SESConv_Z2_H, SESConv_H_H_1x1)):
                nelement = m.weight.nelement()
                n = nelement / m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

class CC_Decoder(nn.Module):
    def __init__(self, matrix_size):
        super(CC_Decoder, self).__init__()
        self.matrix_size = matrix_size
        self.last1 = nn.Linear(self.matrix_size, 1)
        self.act = nn.PReLU()
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        self.act3 = nn.PReLU()
        self.act4 = nn.PReLU()
        self.inr1 = nn.Linear(self.matrix_size, self.matrix_size)
        self.inr2 = nn.Linear(self.matrix_size, self.matrix_size)
        self.inr3 = nn.Linear(self.matrix_size, self.matrix_size)
        self.inr4 = nn.Linear(self.matrix_size, self.matrix_size)
        self._initialize_weights()
        self.kl = 0

    def forward(self, feature_high, x2): # [4, 256, 32, 32], [4, 1024, 256]
        b, n_query_pts = x2.shape[0], x2.shape[1]
        feature_high = F.interpolate(feature_high, size=(64, 64), mode='bicubic', align_corners=True)
        density_w = feature_high.shape[2]
        density_h = feature_high.shape[3]
        b1_f = torch.reshape(feature_high, (b, self.matrix_size, density_w*density_h))
        b1_f = b1_f.transpose(1,2)
        out1 = self.inr1(b1_f)
        out1 = self.act1(out1) + b1_f
        out2 = self.inr2(out1)
        out2 = self.act2(out2) + out1
        out3 = self.inr3(out2)
        out3 = self.act3(out3) + out2
        out4 = self.inr4(out3)
        out4 = self.act4(out4) + out3
        out_mu = torch.squeeze(self.act(self.last1(out4)))
        out_mu = out_mu.reshape((b, 1, density_w, density_h))
        return torch.abs(out_mu), self.kl # [4, 1, 64, 64], [4, 4096], [1]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
   
def build_cc_decoder(matrix_size):
    return CC_Decoder(matrix_size)

def make_layers_4(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.PReLU()]
            else:
                layers += [conv2d, nn.PReLU()]
            in_channels = v
    return nn.Sequential(*layers)

class VGG_Backbone(nn.Module):
    def __init__(self):
        super(VGG_Backbone, self).__init__()
        self.frontend_feat1 = [64, 64]
        self.frontend_feat2 = ['M', 128, 128]
        self.frontend_feat3 = ['M', 256, 256, 256, 256]
        self.frontend_feat4 = ['M', 512, 512, 512, 512]
        self.frontend_feat5 = ['M', 512, 512, 512, 512]
        self.layers = nn.ModuleList([make_layers_vgg(self.frontend_feat1, in_channels = 3), make_layers_vgg(self.frontend_feat2, in_channels = 64),
                                     make_layers_vgg(self.frontend_feat3, in_channels = 128), make_layers_vgg(self.frontend_feat4, in_channels = 256),
                                     make_layers_vgg(self.frontend_feat5, in_channels = 512)])
        scales = [0.8 * 1.11**i for i in range(4)]
        self.sesn1 = nn.Sequential(SESConv_Z2_H(64, 64, 3, 3, scales=scales, padding=1, bias=True, basis_type='A'),
                                   SESMaxProjection(), nn.LeakyReLU(True), nn.BatchNorm2d(64))
        self.sesn2 = nn.Sequential(SESConv_Z2_H(128, 128, 3, 3, scales=scales, padding=1, bias=True, basis_type='A'),
                                   SESMaxProjection(), nn.LeakyReLU(True), nn.BatchNorm2d(128),)
        self.sesn3 = nn.Sequential(SESConv_Z2_H(256, 256, 3, 3, scales=scales, padding=1, bias=True, basis_type='A'),
                                   SESMaxProjection(), nn.LeakyReLU(True), nn.BatchNorm2d(256),)
        self.inc1 = nn.Sequential(SESConv_Z2_H(3, 3, 3, 3, scales=scales, padding=1, bias=True, basis_type='A'),
                                  SESMaxProjection(), nn.LeakyReLU(True), nn.BatchNorm2d(3),)
        self.inc2 = nn.Sequential(SESConv_Z2_H(3, 3, 3, 3, scales=scales, padding=1, bias=True, basis_type='A'),
                                  SESMaxProjection(), nn.LeakyReLU(True), nn.BatchNorm2d(3),)
        n_list = [0, 4, 8, 16, 24]
        mod = models.vgg19(pretrained = True)
        for i in range(5):
            for j in range(len(self.layers[i].state_dict().items())):
                list(self.layers[i].state_dict().items())[j][1].data[:] = list(mod.state_dict().items())[j + n_list[i]][1].data[:]

    def forward(self, x): # [4, 3, 256, 256]
        x = self.inc1(x)
        x = self.inc2(x) + x
        x = self.layers[0](x)
        x = self.sesn1(x) + x
        x = self.layers[1](x)
        x = self.sesn2(x) + x
        x = self.layers[2](x)
        x = self.sesn3(x) + x
        if True:
            l = x.shape[2]
            x = F.interpolate(x, size=(l, l), mode='bicubic', align_corners=False)
        x = self.layers[3](x)
        x = self.layers[4](x) # [4, 512, 16, 16]
        return x
        
def make_layers_vgg(cfg, in_channels = 3, batch_norm=False):
    layers = []
    in_channels = in_channels
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)