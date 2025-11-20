import torch.nn.functional as F
from torch import nn
from typing import List
from timm import create_model
from timm.models import features

class Backbone(nn.Module):
    def __init__(self, name: str, pretrained: bool, out_indices: List[int], train_backbone: bool):
        super(Backbone, self).__init__()
        backbone = create_model(name, pretrained=pretrained, features_only=True, out_indices=out_indices)
        self.train_backbone = train_backbone
        self.backbone = backbone
        self.out_indices = out_indices
        if not self.train_backbone:
            for name, parameter in self.backbone.named_parameters():
                parameter.requires_grad_(False)

    def forward(self, x): # [27, 3, 224, 224]
        x = self.backbone(x)
        for i in range(len(x)):
            x[i] = F.relu(x[i])
        return x # [27, 768, 7, 7]

    @property
    def feature_info(self):
        return features._get_feature_info(self.backbone, out_indices=self.out_indices)

def build_backbone(args):
    if args.type_dataset == 'SENSE':
        backbone = Backbone("convnext_small_384_in22ft1k", True, [3], True)
    elif args.type_dataset == 'HT21':
        backbone = Backbone("convnext_tiny_in22ft1k", True, [3], True)
    else:
        print('This dataset does not exist')
        raise NotImplementedError
    return backbone