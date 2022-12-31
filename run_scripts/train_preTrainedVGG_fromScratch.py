import argparse

import os, sys, itertools
from pathlib import Path
sys.path.append('..')

import torch
from torch import nn, optim
from torch.nn import functional as F

from PreTrainedFIM.model import VGG16_FIM, VGG16bn_FIM
from PreTrainedFIM.util import CIFAR10Worker

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', action='store_true', help='pretrained weight')
parser.add_argument('--weight_fixing', action='store_true', help='gradient False of pretrrained weight')
args = parser.parse_args()
name0 = '_p1' if args.pretrained else '_p0'
name1 = '_w1' if args.weight_fixing else '_w0'
params = {'epoch_num': 10,
          'log_interval': 1250}
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
vgg_model = VGG16bn_FIM(pretrained=args.pretrained, weight_fixing=args.weight_fixing).to(device)
#if torch.cuda.device_count() > 1: 
#    vgg_model = nn.DataParallel(vgg_model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg_model.parameters(), lr=0.0001)
#optimizer = optim.SGD(vgg_model.parameters(), lr=0.01, momentum=0.8, weight_decay=5e-4)
worker = CIFAR10Worker(device, vgg_model, criterion, params)
worker = worker.load_data_loader()

for epoch in range(1, params['epoch_num'] + 1):
    filename = f'{epoch}epoch_VGG16bn' + name0 + name1 + '.pth'
    worker.train(epoch, optimizer)
    worker = worker.set_save_path(filename)
    worker = worker.save_chckpt()