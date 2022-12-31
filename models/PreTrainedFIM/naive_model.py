import itertools
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import numpy as np

class VGG16bn_Naive(nn.Module):
    def __init__(self, pretrained = True, weight_fixing = True, 
                 training = True, num_classes = 10):
        super().__init__()
        self.training = training
        self.pretrained = self._load_pretrained_model(pretrained, weight_fixing)
        self.features = next(itertools.islice(pretrained.children(), 0, 1))
        self.avgpool = next(itertools.islice(pretrained.children(), 1, 2))
        # self.classifier = next(itertools.islice(pretrained.children(), 2, 3))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, num_classes))

    def forward(self,x):
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    def _load_pretrained_model(self, pretrained = True, weight_fixing = True):
        vgg_model = torchvision.models.vgg16_bn(pretrained = pretrained)
        child_counter = 0
        if weight_fixing:
            for child in vgg_model.children():
                for param in child.parameters():
                    param.requires_grad = False
        return vgg_model