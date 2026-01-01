import torch.nn.functional as F
import torch
import torch.nn as nn
from config import cfg
from models.Transformers.ST import ST
from models.FPN.FPN_Head import FPN

backbone = cfg.MODEL_ARCH.backbone

class SCAModule(nn.Module):
    def __init__(self, inn, out):
        super(SCAModule, self).__init__()
        base = inn // 4
        self.conv_sa = nn.Sequential(Conv2d(inn, base, 3, same_padding=True, bias=False), SAM(base),
                                     Conv2d(base, base, 3, same_padding=True, bias=False))
        self.conv_ca = nn.Sequential(Conv2d(inn, base, 3, same_padding=True, bias=False), CAM(),
                                     Conv2d(base, base, 3, same_padding=True, bias=False))
        self.conv_cat = Conv2d(base * 2, out, 1, same_padding=True, bn=False)

    def forward(self, x): # [4, 64, 64, 64]
        sa_feat = self.conv_sa(x)
        ca_feat = self.conv_ca(x)
        cat_feat = torch.cat((sa_feat,ca_feat),1)
        cat_feat = self.conv_cat(cat_feat) # [4, 6, 64, 64]
        return cat_feat   

class SAM(nn.Module):
    def __init__(self, channel):
        super(SAM, self).__init__()
        self.para_lambda = nn.Parameter(torch.zeros(1))
        self.query_conv = Conv2d(channel, channel//8, 1, NL='none')
        self.key_conv = Conv2d(channel, channel//8, 1, NL='none')
        self.value_conv = Conv2d(channel, channel, 1, NL='none')

    def forward(self, x): # [4, 16, 64, 64]
        N, C, H, W = x.size() 
        proj_query = self.query_conv(x).view(N, -1, W * H).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(N, -1, W * H)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy,dim=-1)
        proj_value = self.value_conv(x).view(N, -1, W * H)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(N, C, H, W)
        out = self.para_lambda * out + x # [4, 16, 64, 64]
        return out

class CAM(nn.Module):
    def __init__(self):
        super(CAM, self).__init__()
        self.para_mu = nn.Parameter(torch.zeros(1))

    def forward(self, x): # [4, 16, 64, 64]
        N, C, H, W = x.size() 
        proj_query = x.view(N, C, -1)
        proj_key = x.view(N, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = x.view(N, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(N, C, H, W)
        out = self.para_mu * out + x # [4, 16, 64, 64]
        return out

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=True, bias=True):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True) 
        elif NL == 'prelu':
            self.relu = nn.PReLU() 
        else:
            self.relu = None

    def forward(self, x): # [4, 64, 64, 64]
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x # [4, 16, 64, 64]

class MCC(nn.Module):
    def __init__(self, pretrained=True):
        super(MCC, self).__init__()
        self.fusion = cfg.MM
        if self.fusion:
            in_chans = 4
        else:
            in_chans = 3
        self.Extractor = ST(in_chans=in_chans, embed_dim=backbone['embed_dim'], depths=backbone['depths'], num_heads=backbone['num_heads'], window_size=backbone['window_size'],
                            ape=backbone['ape'], drop_path_rate=backbone['drop_path_rate'], patch_norm=backbone['patch_norm'], use_checkpoint=backbone['use_checkpoint'],
                            out_indices=backbone['out_indices'])
        if pretrained:
            self.Extractor.init_weights(cfg.PRE_WEIGHTS)
        in_channels = cfg.MODEL_ARCH.decode_head.in_channels
        fpn_out_ch = 128
        self.neck = FPN(in_channels, fpn_out_ch, len(in_channels))
        self.project_in = len(in_channels) * fpn_out_ch
        self.project_out = 64
        self.reduce = nn.Conv2d(self.project_in, self.project_out, 1, 1)
        self.output_layer = SCAModule(self.project_out, 6)

    def forward(self, rgb, nir): # [4, 3, 512, 512], [4, 1, 512, 512]
        if self.fusion:
            x = torch.cat([rgb, nir], dim=1)
        else:
            x = rgb
        x = self.Extractor(x)
        x = self.neck(x)
        x = torch.cat([F.interpolate(x[0], scale_factor=0.5), x[1], F.interpolate(x[2], scale_factor=2)], dim=1)
        x = self.reduce(x)
        x = self.output_layer(x) # [4, 6, 64, 64]
        return x