#!/usr/bin/env python37
# -*- coding: utf-8 -*-
import torch
from torch import nn
import torchvision.models as models

class UniImageResNet50(nn.Module):
    def __init__(self, img_ptm_path, num_classes):
        super(UniImageResNet50, self).__init__()
        resnet = models.resnet50()
        resnet.load_state_dict(torch.load(img_ptm_path))
        # strip the last layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size = 1)
        self.classifier = nn.Linear(2048, num_classes, bias = False)

    def forward(self, imgs):
        output = self.feature_extractor(imgs)
        output = self.avgpool(output)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)

        return output



class ResNet101ImageEncoder(nn.Module):
    def __init__(self, img_ptm_path):
        super(ResNet101ImageEncoder, self).__init__()
        resnet = models.resnet101()
        resnet.load_state_dict(torch.load(img_ptm_path))
        # strip the last layer
        self.feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, imgs):
        output = self.feature_extractor(imgs)

        return output


class ResNet152ImageEncoder(nn.Module):
    def __init__(self, img_ptm_path, pool_type = 'avg', num_output_features = 1):
        super(ResNet152ImageEncoder, self).__init__()
        resnet152 = models.resnet152()
        resnet152.load_state_dict(torch.load(img_ptm_path))
        modules = list(resnet152.children())[:-2]
        self.model = nn.Sequential(*modules)

        pool_func = (
            nn.AdaptiveAvgPool2d if pool_type == "avg" else nn.AdaptiveMaxPool2d
        )

        # -1 will keep the original feature size
        if num_output_features == -1:
            self.pool = nn.Identity()
        elif num_output_features in [1, 2, 3, 5, 7]:
            self.pool = pool_func((num_output_features, 1))
        elif num_output_features == 4:
            self.pool = pool_func((2, 2))
        elif num_output_features == 6:
            self.pool = pool_func((3, 2))
        elif num_output_features == 8:
            self.pool = pool_func((4, 2))
        elif num_output_features == 9:
            self.pool = pool_func((3, 3))

        self.out_dim = 2048

    def forward(self, x):
        # Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048
        out = self.pool(self.model(x))
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out

    
class UniImageRes101(nn.Module):
    def __init__(self, img_ptm_path, img_ft_dim, num_classes):
        super(UniImageRes101, self).__init__()
        self.img_ft_extractor = ResNet101ImageEncoder(img_ptm_path)
        self.classifier = nn.Linear(img_ft_dim, num_classes, bias = True)

    def forward(self, imgs):
        img_feat = self.img_ft_extractor(imgs)
        img_feat = img_feat.squeeze(2).squeeze(2)
        output = self.classifier(img_feat)
        return output


class UniImageRes152(nn.Module):
    def __init__(self, img_ptm_path, img_ft_dim, num_classes, num_output_features = 1):
        super(UniImageRes152, self).__init__()
        self.img_ft_extractor = ResNet152ImageEncoder(img_ptm_path, num_output_features = num_output_features)
        cls_in_dim = img_ft_dim * num_output_features
        self.classifier = nn.Linear(cls_in_dim, num_classes)

    def forward(self, imgs):
        img_feat = self.img_ft_extractor(imgs)
        img_feat = img_feat.squeeze(1)
        output = self.classifier(img_feat)
        return output
