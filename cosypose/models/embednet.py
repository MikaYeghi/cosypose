import torch
import torch.nn as nn
import torch.optim as optim
from os import name
from torchvision.models import ResNet
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.resnet import BasicBlock
import cv2
from glob import glob
import os
from matplotlib import pyplot as plt
import pdb
from tqdm import tqdm
import torchvision.transforms as transforms


class EmbeddingsNetwork(ResNet):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super().__init__(block, layers, num_classes=num_classes, zero_init_residual=zero_init_residual, groups=groups, width_per_group=width_per_group, replace_stride_with_dilation=replace_stride_with_dilation, norm_layer=norm_layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        # Remove average pooling from the original network definition in order to slice through the last layer which generates features
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

    def get_preprocessing(self):
        """
        All torchvision models have the same preprocessing.
        Taken from:
        https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683
        """
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        return normalize

    def preprocess(self, batch_images):
        preprocessing = self.get_preprocessing()
        return preprocessing(batch_images)

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
        'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
        'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
        'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
    }

    model = EmbeddingsNetwork(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def embednet(backbone='resnet18', pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # Extract layers
    if backbone == 'resnet18':
        layers = [2, 2, 2, 2]
    elif backbone == 'resnet34':
        layers = [3, 4, 6, 3]
    else:
        NotImplementedError

    return _resnet(backbone, BasicBlock, layers, pretrained, progress,
                   **kwargs)