import itertools
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import numpy as np
from .reparameterize_layer import ReparamterNorm, MultiSequential

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

input_dims_cfgs = {
    'D': [32, 32, 'M', 16, 16, 'M', 8, 8, 8, 'M', 4, 4, 4, 'M', 2, 2, 2, 'M']
}

class VGG16bn_FIM(nn.Module):
    def __init__(self, pretrained = True, weight_fixing = True, 
                 num_classes = 10):
        super().__init__()

        pretrained = self._load_pretrained_model(pretrained, weight_fixing)
        child = next(itertools.islice(pretrained.children(), 0, 1))
        cfgs_pair = zip(cfgs['D'], input_dims_cfgs['D'])
        self.features = load_pretrained_with_repara(cfgs_pair, child, batch_norm=True)
        self.avgpool = next(itertools.islice(pretrained.children(), 1, 2))
        # self.classifier = next(itertools.islice(pretrained.children(), 2, 3))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, num_classes))
        self.evaluate_FIM_mode = False

        # constant in KL divergence
        self.w_squared = torch.tensor(0, requires_grad=False).float()
        self.layer_dims = torch.zeros(1, 13, requires_grad=False).float()
        layer_id = 0
        for _, layer in enumerate(self.features.children()):
            if isinstance(layer, nn.Conv2d):
                params = next(layer.parameters())
                self.w_squared = self.w_squared + params.pow(2).sum()
                self.layer_dims[0,layer_id] = torch.tensor(np.prod(params.size()))
                layer_id += 1

    def forward(self,x):
        x, y = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, y

    def _load_pretrained_model(self, pretrained = True, weight_fixing = True):
        vgg_model = torchvision.models.vgg16_bn(pretrained = pretrained)
        child_counter = 0
        if weight_fixing:
            for child in vgg_model.children():
                for param in child.parameters():
                    param.requires_grad = False
        return vgg_model

    def to_device_child_tensors(self, device):
        self.layer_dims = self.layer_dims.to(device)
        self.w_squared = self.w_squared.to(device)
        for _, layer in enumerate(self.features.children()):
            if isinstance(layer, ReparamterNorm):
                layer.logvar = layer.logvar.to(device)
        return self

    def eval_FIM(self):
        self.evaluate_FIM_mode = True
        self._switch_parameter_gradient()

    def not_eval_FIM(self):
        self.evaluate_FIM_mode = False
        self._switch_parameter_gradient()
                
    def _switch_parameter_gradient(self):
        for param in self.parameters():
            param.requires_grad = not self.evaluate_FIM_mode
        for _, layer in enumerate(self.features.children()):
            if isinstance(layer, ReparamterNorm):
                layer.reparameterization = self.evaluate_FIM_mode
                for param in layer.logvar.parameters():
                    param.requires_grad = self.evaluate_FIM_mode

    def inactivate_parameters_ex_specific_layer(self, layerId):
        """
        for VGG16, layerId is from 0 up to 12
        """
        count_ReparameterNorm = 0
        for _, layer in enumerate(self.features.children()):
            if isinstance(layer, ReparamterNorm):
                if count_ReparameterNorm == layerId:
                    for param in layer.logvar.parameters():
                        param.requires_grad = True
                    layer.reparameterization = True
                else:
                    for param in layer.logvar.parameters():
                        param.requires_grad = False
                    layer.reparameterization = False
                count_ReparameterNorm += 1

    def initialize_FIM_weight(self):
        def init_weights(m):
            if type(m) == nn.Conv2d:
                nn.init.normal_(m.weight, mean=0, std=0.01)
                m.bias.data.fill_(0.01)
        
        for layer in self.features:
            if isinstance(layer, ReparamterNorm):
                print(layer.reparameterization)
                layer.logvar.apply(init_weights)

"""
def make_layers_with_repara(cfg, batch_norm=False):
    layers_ctr = 0
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            layers_ctr += 1
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            repara = ReparamterNorm()
            if batch_norm:
                layers += [conv2d, repara, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                layers_ctr += 3
            else:
                layers += [conv2d, repara, nn.ReLU(inplace=True)]
                layers_ctr += 2
            in_channels = v
    return nn.Sequential(*layers)
"""

def load_pretrained_with_repara(cfgs_pair, child, batch_norm=False):
    layer_ctr = 0
    layers = []
    layer_contrasts = []
    in_channels = 3
    for v, u in cfgs_pair:
        if v == 'M':
            layers += [child[layer_ctr]]
            layer_contrasts += [0]
            layer_ctr += 1
        else:
            conv2d = child[layer_ctr]
            repara = ReparamterNorm(input_dims = (v, u, u), hidden_dim = v)
            if batch_norm:
                layers += [conv2d, repara, child[layer_ctr+1], child[layer_ctr+2]]
                layer_contrasts += [0, 1, 0, 0]
                layer_ctr += 3
            else:
                layers += [conv2d, repara, child[layer_ctr]]
                layer_contrasts += [0, 1, 0]
                layer_ctr += 2
    return MultiSequential(layer_contrasts, *layers)
    #, nn.Parameter(torch.squeeze(torch.stack(logvarslist)))