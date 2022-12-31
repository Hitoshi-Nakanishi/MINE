import os, sys, itertools
from pathlib import Path
sys.path.append('..')

import torch
from torch import nn, optim
from torch.nn import functional as F

from PreTrainedFIM.model import VGG16_FIM, VGG16bn_FIM
from PreTrainedFIM.util import CIFAR10Worker

params = {'epoch_num': 10,
          'log_interval': 1250}
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

vgg_model = VGG16bn_FIM().to(device)
#if torch.cuda.device_count() > 1: 
#    vgg_model = nn.DataParallel(vgg_model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg_model.parameters(), lr=1e-3)
worker = CIFAR10Worker(device, vgg_model, criterion, params)
worker = worker.load_data_loader()

for epoch in range(1, params['epoch_num'] + 1):
    worker.train(epoch, optimizer)
    worker = worker.set_save_path(f'{epoch}epoch_VGG16bn.pth')
    worker = worker.save_chckpt()