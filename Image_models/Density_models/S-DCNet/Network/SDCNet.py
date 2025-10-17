import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F
import math
from Network.class_func import Class2Count
from Network.merge_func import count_merge_low2high_batch

def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
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
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)   

class one_conv(nn.Module):
    def __init__(self, in_ch, out_ch, normaliz=False):
        super(one_conv, self).__init__()
        ops = []
        ops += [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
        if normaliz:
            ops += [nn.BatchNorm2d(out_ch)]
        ops += [nn.ReLU(inplace=True)]
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch, normaliz=False):
        super(double_conv, self).__init__()
        ops = []
        ops += [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
        if normaliz:
            ops += [nn.BatchNorm2d(out_ch)]
        ops += [nn.ReLU(inplace=True)]
        ops += [nn.Conv2d(out_ch, out_ch, 3, padding=1)]
        if normaliz:
            ops += [nn.BatchNorm2d(out_ch)]
        ops += [nn.ReLU(inplace=True)]
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, up_in_ch, up_out_ch,cat_in_ch, cat_out_ch,if_convt=False):
        super(up, self).__init__()
        self.if_convt = if_convt
        if self.if_convt:
            self.up = nn.ConvTranspose2d(up_in_ch,up_out_ch, 2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.conv1 = one_conv(up_in_ch,up_out_ch)
        self.conv2 = double_conv(cat_in_ch, cat_out_ch)

    def forward(self, x1, x2):
        if self.if_convt:
            x1 = self.up(x1)
        else:
            x1 = self.up(x1)
            x1 = self.conv1(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, int(math.ceil(diffX / 2.0)), diffY // 2, int(math.ceil(diffY / 2.0))))
        x = torch.cat([x2, x1], dim=1)
        del x2,x1
        x = self.conv2(x)
        return x

class SDCNet_VGG16_classify(nn.Module):
    def __init__(self, class_num, label_indice, div_times=2, load_weights=False, freeze_bn=False, psize=64, pstride=64, IF_pre_bn=True, parse_method ='maxp', merge_reso='low'):
        super(SDCNet_VGG16_classify, self).__init__()
        self.label_indice = label_indice
        self.class_num = len(self.label_indice) + 1
        self.div_times = div_times
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512,'M']
        self.args = dict()
        self.args['label_indice'] = self.label_indice
        self.args['class_num'] = self.class_num
        self.args['psize'],self.args['pstride'] =  psize,pstride
        self.args['div_times'] = self.div_times
        self.args['frontend'] = self.frontend_feat
        self.conv1_features = make_layers([64, 64, 'M'], in_channels=3)
        self.conv2_features = make_layers([128, 128, 'M'], in_channels=64)
        self.conv3_features = make_layers([256, 256, 256, 'M'], in_channels=128)
        self.conv4_features = make_layers([512, 512, 512,'M'], in_channels=256)
        self.conv5_features = make_layers([512, 512, 512,'M'], in_channels=512)
        self.fc = torch.nn.Sequential(torch.nn.AvgPool2d((2, 2), stride=2), torch.nn.ReLU(),
                                      torch.nn.Conv2d(512, 512, (1, 1)), torch.nn.ReLU(),
                                      torch.nn.Conv2d(512, class_num, (1, 1)))
        self.up45 = up(up_in_ch=512, up_out_ch=256, cat_in_ch=(256 + 512), cat_out_ch=512)
        self.lw_fc = torch.nn.Sequential(torch.nn.AvgPool2d((2, 2), stride=2), torch.nn.ReLU(),
                                         torch.nn.Conv2d(512, 512, (1, 1), stride=1), torch.nn.ReLU(),
                                         torch.nn.Conv2d(512, 1, (1, 1)))
        self.up34 = up(up_in_ch=512, up_out_ch=256, cat_in_ch=(256 + 256), cat_out_ch=512)
        self._initialize_weights()
        if load_weights:
            mod = models.vgg16(pretrained = True)
            pretrained_dict = mod.state_dict()
            net_dict = self.state_dict()
            net_dict_name = self.state_dict().keys()
            lay_num = 0
            for name, params in list(pretrained_dict.items()):                                                   
                if 'conv' in list(net_dict_name)[lay_num]:
                    net_dict[list(net_dict_name)[lay_num]] = pretrained_dict[name]
                    lay_num = lay_num + 1
                else:
                    break
            self.load_state_dict(net_dict)
   
    def forward(self, x): # [1, 3, 704, 1088]
        x = self.conv1_features(x) # [1, 64, 352, 544]
        x = self.conv2_features(x) # [1, 128, 176, 272]
        x = self.conv3_features(x) # [1, 256, 88, 136]
        conv3_feat = x if self.div_times > 1 else []
        x = self.conv4_features(x) # [1, 512, 44, 68]
        conv4_feat = x if self.div_times > 0 else []
        x = self.conv5_features(x) # [1, 512, 22, 34]
        conv5_feat = x if self.div_times > 0 else []
        x = self.fc(x) # [1, 55, 11, 17]
        feature_map = {'conv3': conv3_feat, 'conv4': conv4_feat, 'conv5': conv5_feat, 'cls0': x}
        return feature_map

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def resample(self,feature_map):
        div_res = dict()
        div_res['cls0'] = feature_map['cls0']
        if self.div_times > 0:
            new_conv4 = self.up45(feature_map['conv5'],feature_map['conv4'])
            new_conv4_w = self.lw_fc(new_conv4)
            new_conv4_w = torch.sigmoid(new_conv4_w)
            new_conv4_reg = self.fc(new_conv4)
            del feature_map['conv5'],feature_map['conv4']
            div_res['cls1'] = new_conv4_reg
            div_res['w1'] = 1-new_conv4_w
        if  self.div_times > 1:
            new_conv3 = self.up34(new_conv4,feature_map['conv3'])
            new_conv3_w = self.lw_fc(new_conv3)
            new_conv3_w = torch.sigmoid(new_conv3_w)
            new_conv3_reg = self.fc(new_conv3)
            del feature_map['conv3']
            del new_conv3,new_conv4
            div_res['cls2'] = new_conv3_reg
            div_res['w2'] = 1-new_conv3_w
        feature_map['cls0']=[]
        del feature_map
        return div_res

    def parse_merge(self,div_res):
        res = dict()
        for cidx in range(self.div_times+1):
            tname = 'c' + str(cidx)
            div_res['cls'+str(cidx)] = div_res['cls' + str(cidx)].max(dim=1,keepdim=True)[1]
            res[tname] = Class2Count(div_res['cls' + str(cidx)], self.label_indice)
        res['div0'] = res['c0'] 
        for divt in range(1,self.div_times+1):
            tname = 'div' + str(divt) 
            tchigh = res['c' + str(divt)] 
            tclow = res['div' + str(int(divt-1))]
            tclow = count_merge_low2high_batch(tclow,tchigh)
            tw = div_res['w'+str(divt)]
            res[tname] = (1 - tw) * tclow + tw * tchigh
        del div_res
        return res