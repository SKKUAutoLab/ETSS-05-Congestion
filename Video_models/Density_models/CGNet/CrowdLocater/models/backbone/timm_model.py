from torch import nn
from typing import Dict, List
from timm import create_model
from timm.models import features

class Backbone(nn.Module):
    def __init__(self, name: str,pretrained:bool,out_indices:List[int], train_backbone: bool,others:Dict):
        super(Backbone,self).__init__()
        backbone = create_model(name,pretrained=pretrained,features_only=True, out_indices=out_indices,**others)
        self.train_backbone = train_backbone
        self.backbone = backbone
        self.out_indices = out_indices
        if not self.train_backbone:
            for name, parameter in self.backbone.named_parameters():
                parameter.requires_grad_(False)

    def forward(self,x):
        return self.backbone(x)
    
    @property
    def feature_info(self):
        return features._get_feature_info(self.backbone,out_indices=self.out_indices)

def build_backbone(args):
    backbone = Backbone(args.name, args.pretrained, args.out_indices, args.train_backbone, args.others)
    return backbone