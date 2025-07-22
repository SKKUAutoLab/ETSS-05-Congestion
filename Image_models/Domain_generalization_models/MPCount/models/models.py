import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from math import sqrt
import functools

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, bn=False, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x): # [16, 512, 20, 20]
        y = self.conv(x) # [16, 1024, 20, 20]
        if self.bn is not None:
            y = self.bn(y) # [16, 1024, 20, 20]
        if self.relu is not None:
            y = self.relu(y) # [16, 1024, 20, 20]
        return y

# https://github.com/open-mmlab/mmsegmentation/issues/255
class FixedUpsample(nn.Module):
    def __init__(self, channel: int, scale_factor: int):
        super().__init__()
        assert isinstance(scale_factor, int) and scale_factor > 1 and scale_factor % 2 == 0
        self.scale_factor = scale_factor
        kernel_size = scale_factor + 1
        self.weight = nn.Parameter(torch.empty((1, 1, kernel_size, kernel_size), dtype=torch.float32).expand(channel, -1, -1, -1).clone())
        self.conv = functools.partial(F.conv2d, weight=self.weight, bias=None, padding=scale_factor // 2, groups=channel)
        with torch.no_grad():
            self.weight.fill_(1 / (kernel_size * kernel_size))

    def forward(self, t): # [16, 512, 20, 20]
        if t is None:
            return t
        return self.conv(F.interpolate(t, scale_factor=self.scale_factor, mode='nearest')) # [16, 512, 40, 40]
    
class Upsample(nn.Module):
    def __init__(self, channel: int, scale_factor: int, deterministic=True):
        super().__init__()
        self.deterministic = deterministic
        if deterministic:
            self.upsample = FixedUpsample(channel, scale_factor)
        else:
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

    def forward(self, t): # [16, 512, 20, 20]
        return self.upsample(t) # [16, 512, 40, 40]
    
def upsample(x, scale_factor=2, mode='bilinear'): # [16, 1, 20, 20]
    if mode == 'nearest':
        return F.interpolate(x, scale_factor=scale_factor, mode=mode) # [16, 1, 80, 80]
    else:
        return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=False)
    
class DGModel_base(nn.Module):
    def __init__(self, pretrained=True, den_dropout=0.5, deterministic=True):
        super().__init__()
        self.den_dropout = den_dropout
        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None)
        # vgg encoder blocks
        self.enc1 = nn.Sequential(*list(vgg.features.children())[:23])
        self.enc2 = nn.Sequential(*list(vgg.features.children())[23:33])
        self.enc3 = nn.Sequential(*list(vgg.features.children())[33:43])
        # decoder blocks
        self.dec3 = nn.Sequential(ConvBlock(512, 1024, bn=True), ConvBlock(1024, 512, bn=True))
        self.dec2 = nn.Sequential(ConvBlock(1024, 512, bn=True), ConvBlock(512, 256, bn=True))
        self.dec1 = nn.Sequential(ConvBlock(512, 256, bn=True), ConvBlock(256, 128, bn=True))
        self.den_dec = nn.Sequential(ConvBlock(512 + 256 + 128, 256, kernel_size=1, padding=0, bn=True), nn.Dropout2d(p=den_dropout))
        self.den_head = nn.Sequential(ConvBlock(256, 1, kernel_size=1, padding=0))
        self.upsample1 = Upsample(512, 2, deterministic)
        self.upsample2 = Upsample(256, 2, deterministic)
        self.upsample3 = Upsample(256, 2, deterministic)
        self.upsample4 = Upsample(512, 4, deterministic)
        self.upsample_d = Upsample(1, 4, deterministic)

    def forward_fe(self, x): # [16, 3, 320, 320]
        x1 = self.enc1(x) # [16, 256, 80, 80]
        x2 = self.enc2(x1) # [16, 512, 40, 40]
        x3 = self.enc3(x2) # [16, 512, 20, 20]
        x = self.dec3(x3) # [16, 512, 20, 20]
        y3 = x # [16, 512, 20, 20]
        x = self.upsample1(x) # [16, 512, 40, 40]
        x = torch.cat([x, x2], dim=1) # [16, 1024, 40, 40]
        x = self.dec2(x) # [16, 256, 40, 40]
        y2 = x # [16, 256, 40, 40]
        x = self.upsample2(x) # [16, 256, 80, 80]
        x = torch.cat([x, x1], dim=1) # [16, 512, 80, 80]
        x = self.dec1(x) # [16, 128, 80, 80]
        y1 = x # [16, 128, 80, 80]
        y2 = self.upsample3(y2) # [16, 256, 80, 80]
        y3 = self.upsample4(y3) # [16, 512, 80, 80]
        y_cat = torch.cat([y1, y2, y3], dim=1) # [16, 896, 80, 80]
        return y_cat, x3
    
    def forward(self, x): # [16, 3, 320, 320]
        y_cat, _ = self.forward_fe(x) # [16, 896, 80, 80]
        y_den = self.den_dec(y_cat) # [16, 256, 80, 80]
        d = self.den_head(y_den) # [16, 1, 80, 80]
        d = self.upsample_d(d) # [16, 1, 320, 320]
        return d
    
class DGModel_mem(DGModel_base):
    def __init__(self, pretrained=True, mem_size=1024, mem_dim=256, den_dropout=0.5, deterministic=True):
        super().__init__(pretrained, den_dropout, deterministic)
        self.mem_size = mem_size
        self.mem_dim = mem_dim
        self.mem = nn.Parameter(torch.FloatTensor(1, self.mem_dim, self.mem_size).normal_(0.0, 1.0))
        self.den_dec = nn.Sequential(ConvBlock(512 + 256 + 128, self.mem_dim, kernel_size=1, padding=0, bn=True), nn.Dropout2d(p=den_dropout))
        self.den_head = nn.Sequential(ConvBlock(self.mem_dim, 1, kernel_size=1, padding=0))

    def forward_mem(self, y): # [16, 256, 80, 80]
        b, k, h, w = y.shape
        m = self.mem.repeat(b, 1, 1) # [16, 256, 1024]
        m_key = m.transpose(1, 2) # [16, 1024, 256]
        y_ = y.view(b, k, -1) # [16, 256, 6400]
        logits = torch.bmm(m_key, y_) / sqrt(k) # [16, 1024, 6400]
        y_new = torch.bmm(m_key.transpose(1, 2), F.softmax(logits, dim=1)) # [16, 256, 6400]
        y_new_ = y_new.view(b, k, h, w) # [16, 256, 80, 80]
        return y_new_, logits
    
    def forward(self, x):
        y_cat, _ = self.forward_fe(x)
        y_den = self.den_dec(y_cat)
        y_den_new, _ = self.forward_mem(y_den)
        d = self.den_head(y_den_new)
        d = self.upsample_d(d)
        return d
    
class DGModel_memadd(DGModel_mem):
    def __init__(self, pretrained=True, mem_size=1024, mem_dim=256, den_dropout=0.5, err_thrs=0.5, deterministic=True):
        super().__init__(pretrained, mem_size, mem_dim, den_dropout, deterministic)
        self.err_thrs = err_thrs
        self.den_dec = nn.Sequential(ConvBlock(512+256+128, 256, kernel_size=1, padding=0, bn=True))

    def jsd(self, logits1, logits2):
        p1 = F.softmax(logits1, dim=1)
        p2 = F.softmax(logits2, dim=1)
        jsd = F.mse_loss(p1, p2)
        return jsd

    def forward_train(self, img1, img2):
        y_cat1, _ = self.forward_fe(img1)
        y_cat2, _ = self.forward_fe(img2)
        y_den1 = self.den_dec(y_cat1)
        y_den2 = self.den_dec(y_cat2)
        y_in1 = F.instance_norm(y_den1, eps=1e-5)
        y_in2 = F.instance_norm(y_den2, eps=1e-5)
        e_y = torch.abs(y_in1 - y_in2)
        e_mask = (e_y < self.err_thrs).clone().detach()
        y_den_masked1 = F.dropout2d(y_den1 * e_mask, self.den_dropout)
        y_den_masked2 = F.dropout2d(y_den2 * e_mask, self.den_dropout)
        y_den_new1, logits1 = self.forward_mem(y_den_masked1)
        y_den_new2, logits2 = self.forward_mem(y_den_masked2)
        loss_con = self.jsd(logits1, logits2)
        d1 = self.den_head(y_den_new1)
        d2 = self.den_head(y_den_new2)
        d1 = self.upsample_d(d1)
        d2 = self.upsample_d(d2)
        return d1, d2, loss_con
    
class DGModel_cls(DGModel_base):
    def __init__(self, pretrained=True, den_dropout=0.5, cls_dropout=0.3, cls_thrs=0.5, deterministic=True):
        super().__init__(pretrained, den_dropout, deterministic)
        self.cls_dropout = cls_dropout
        self.cls_thrs = cls_thrs
        self.cls_head = nn.Sequential(ConvBlock(512, 256, bn=True), nn.Dropout2d(p=self.cls_dropout),
                                      ConvBlock(256, 1, kernel_size=1, padding=0, relu=False), nn.Sigmoid())

    def transform_cls_map_gt(self, c_gt):
        return upsample(c_gt, scale_factor=4, mode='nearest')
    
    def transform_cls_map_pred(self, c):
        c_new = c.clone().detach()
        c_new[c < self.cls_thrs] = 0
        c_new[c >= self.cls_thrs] = 1
        c_resized = upsample(c_new, scale_factor=4, mode='nearest')
        return c_resized

    def transform_cls_map(self, c, c_gt=None):
        if c_gt is not None:
            return self.transform_cls_map_gt(c_gt)
        else:
            return self.transform_cls_map_pred(c)
    
    def forward(self, x, c_gt=None):
        y_cat, x3 = self.forward_fe(x)
        y_den = self.den_dec(y_cat)
        c = self.cls_head(x3)
        c_resized = self.transform_cls_map(c, c_gt)
        d = self.den_head(y_den)
        dc = d * c_resized
        dc = self.upsample_d(dc)
        return dc, c
    
class DGModel_memcls(DGModel_mem):
    def __init__(self, pretrained=True, mem_size=1024, mem_dim=256, den_dropout=0.5, cls_dropout=0.3, cls_thrs=0.5, deterministic=True):
        super().__init__(pretrained, mem_size, mem_dim, den_dropout, deterministic)
        self.cls_dropout = cls_dropout
        self.cls_thrs = cls_thrs
        self.cls_head = nn.Sequential(ConvBlock(512, 256, bn=True), nn.Dropout2d(p=self.cls_dropout),
                                      ConvBlock(256, 1, kernel_size=1, padding=0, relu=False), nn.Sigmoid())

    def transform_cls_map_gt(self, c_gt): # [16, 1, 20, 20]
        return upsample(c_gt, scale_factor=4, mode='nearest') # [16, 1, 80, 80]
    
    def transform_cls_map_pred(self, c): # [16, 1, 20, 20]
        c_new = c.clone().detach()
        c_new[c < self.cls_thrs] = 0
        c_new[c >= self.cls_thrs] = 1
        c_resized = upsample(c_new, scale_factor=4, mode='nearest') # [16, 1, 80, 80]
        return c_resized

    def transform_cls_map(self, c, c_gt=None): # [1, 1, 48, 64], None
        if c_gt is not None:
            return self.transform_cls_map_gt(c_gt)
        else:
            return self.transform_cls_map_pred(c)
    
    def forward(self, x, c_gt=None): # [1, 3, 768, 1024], None
        y_cat, x3 = self.forward_fe(x) # [1, 896, 192, 256], [1, 512, 48, 64]
        y_den = self.den_dec(y_cat) # [1, 256, 192, 256]
        y_den_new, _ = self.forward_mem(y_den) # [1, 256, 192, 256]
        c = self.cls_head(x3) # [1, 1, 48, 64]
        c_resized = self.transform_cls_map(c, c_gt) # [1, 1, 192, 256]
        d = self.den_head(y_den_new) # [1, 1, 192, 256]
        dc = d * c_resized # [1, 1, 192, 256]
        dc = self.upsample_d(dc) # [1, 1, 768, 1024]
        return dc, c
    
class DGModel_final(DGModel_memcls):
    def __init__(self, pretrained=True, mem_size=1024, mem_dim=256, cls_thrs=0.5, err_thrs=0.5, den_dropout=0.5, cls_dropout=0.3, has_err_loss=False, deterministic=True):
        super().__init__(pretrained, mem_size, mem_dim, den_dropout, cls_dropout, cls_thrs, deterministic)
        self.err_thrs = err_thrs
        self.has_err_loss = has_err_loss
        self.den_dec = nn.Sequential(ConvBlock(512 + 256 + 128, self.mem_dim, kernel_size=1, padding=0, bn=True))
    
    def jsd(self, logits1, logits2): # [16, 1024, 6400], [16, 1024, 6400]
        p1 = F.softmax(logits1, dim=1) # [16, 1024, 6400]
        p2 = F.softmax(logits2, dim=1) # [16, 1024, 6400]
        jsd = F.mse_loss(p1, p2)
        return jsd
    
    def forward_train(self, img1, img2, c_gt=None): # [16, 3, 320, 320], [16, 3, 320, 320], [16, 1, 20, 20]
        y_cat1, x3_1 = self.forward_fe(img1) # [16, 896, 80, 80], [16, 512, 20, 20]
        y_cat2, x3_2 = self.forward_fe(img2) # [16, 896, 80, 80], [16, 512, 20, 20]
        y_den1 = self.den_dec(y_cat1) # [16, 256, 80, 80]
        y_den2 = self.den_dec(y_cat2) # [16, 256, 80, 80]
        y_in1 = F.instance_norm(y_den1, eps=1e-5) # [16, 256, 80, 80]
        y_in2 = F.instance_norm(y_den2, eps=1e-5) # [16, 256, 80, 80]
        e_y = torch.abs(y_in1 - y_in2) # [16, 256, 80, 80]
        e_mask = (e_y < self.err_thrs).clone().detach() # [16, 256, 80, 80]
        loss_err = F.l1_loss(y_in1, y_in2) if self.has_err_loss else 0
        y_den_masked1 = F.dropout2d(y_den1 * e_mask, self.den_dropout) # [16, 256, 80, 80]
        y_den_masked2 = F.dropout2d(y_den2 * e_mask, self.den_dropout) # [16, 256, 80, 80]
        y_den_new1, logits1 = self.forward_mem(y_den_masked1) # [16, 256, 80, 80], [16, 1024, 6400]
        y_den_new2, logits2 = self.forward_mem(y_den_masked2) # [16, 256, 80, 80], [16, 1024, 6400]
        loss_con = self.jsd(logits1, logits2)
        c1 = self.cls_head(x3_1) # [16, 1, 20, 20]
        c2 = self.cls_head(x3_2) # [16, 1, 20, 20]
        c_resized_gt = self.transform_cls_map_gt(c_gt) # [16, 1, 80, 80]
        c_resized1 = self.transform_cls_map_pred(c1) # [16, 1, 80, 80]
        c_resized2 = self.transform_cls_map_pred(c2) # [16, 1, 80, 80]
        c_err = torch.abs(c_resized1 - c_resized2) # [16, 1, 80, 80]
        c_resized = torch.clamp(c_resized_gt + c_err, 0, 1) # [16, 1, 80, 80]
        d1 = self.den_head(y_den_new1) # [16, 1, 80, 80]
        d2 = self.den_head(y_den_new2) # [16, 1, 80, 80]
        dc1 = self.upsample_d(d1 * c_resized) # [16, 1, 320, 320]
        dc2 = self.upsample_d(d2 * c_resized) # [16, 1, 320, 320]
        c_err = upsample(c_err, scale_factor=4) # [16, 1, 320, 320]
        return dc1, dc2, c1, c2, c_err, loss_con, loss_err

if __name__ == '__main__':
    base_model = DGModel_base()
    input = torch.rand(16, 3, 320, 320)
    output = base_model(input)