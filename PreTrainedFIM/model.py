import itertools
import torch
from torch import nn
import torchvision

class VGG16_FIM(nn.Module):
    def __init__(self):
        super(VGG16_FIM, self).__init__()
        self.pretrained = pretrained = self._load_pretrained_model()
        self.features = child = next(itertools.islice(pretrained.children(), 0, 1))
        self.avgpool = next(itertools.islice(pretrained.children(), 1, 2))
        self.classifier = next(itertools.islice(pretrained.children(), 2, 3))
        self.logvars = torch.FloatTensor(13, 1)
        self.Conv0 = child[0]    # (3,64)
        self.Conv2 = child[2]    # (64,64)
        self.Conv5 = child[5]    # (64,128)
        self.Conv7 = child[7]    # (128,128)
        self.Conv10 = child[10]  # (128,256)
        self.Conv12 = child[12]  # (256,256)
        self.Conv14 = child[14]  # (256,256)
        self.Conv17 = child[17]  # (256,512)
        self.Conv19 = child[19]  # (512,512)
        self.Conv21 = child[21]  # (512,512)
        self.Conv24 = child[24]  # (512,512)
        self.Conv26 = child[26]  # (512,512)
        self.Conv28 = child[28]  # (512,512)

    def forward(self,x):
        out = self.reparameterize(self.Conv0(x), self.logvars[0])
        out = F.relu(out)
        out = self.reparameterize(self.Conv2(out), self.logvars[1])
        out = F.max_pool2d(F.relu(out), 2, 2)
        out = self.reparameterize(self.Conv5(out), self.logvars[2])
        out = F.relu(out)
        out = self.reparameterize(self.Conv7(out), self.logvars[3])
        out = F.max_pool2d(F.relu(out), 2, 2)
        out = self.reparameterize(self.Conv10(out), self.logvars[4])
        out = F.relu(out)
        out = self.reparameterize(self.Conv12(out), self.logvars[5])
        out = F.relu(out)
        out = self.reparameterize(self.Conv14(out), self.logvars[6])
        out = F.max_pool2d(F.relu(out), 2, 2)
        out = self.reparameterize(self.Conv17(out), self.logvars[7])
        out = F.relu(out)
        out = self.reparameterize(self.Conv19(out), self.logvars[8])
        out = F.relu(out)
        out = self.reparameterize(self.Conv21(out), self.logvars[9])
        out = F.max_pool2d(F.relu(out), 2, 2)
        out = self.reparameterize(self.Conv24(out), self.logvars[10])
        out = F.relu(out)
        out = self.reparameterize(self.Conv26(out), self.logvars[11])
        out = F.relu(out)
        out = self.reparameterize(self.Conv28(out), self.logvars[12])
        out = F.max_pool2d(F.relu(out), 2, 2)
        out = self.avgpool(out)
        out = torch.flatten(x, 1)
        out = self.classifier(out)
        return out

    def _load_pretrained_model(self):
        vgg_model = torchvision.models.vgg16(pretrained=True)
        child_counter = 0
        for child in vgg_model.children():
            for param in child.parameters():
                param.requires_grad = False
        return vgg_model
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(mu)
        return mu + eps*std

class VGG16bn_FIM(nn.Module):
    def __init__(self):
        super(VGG16bn_FIM, self).__init__()
        self.pretrained = pretrained = self._load_pretrained_model()
        self.features = child = next(itertools.islice(pretrained.children(), 0, 1))
        self.avgpool = next(itertools.islice(pretrained.children(), 1, 2))
        self.classifier = next(itertools.islice(pretrained.children(), 2, 3))
        self.logvars = torch.ones(1, 13, requires_grad=True) * 0.10
        self.layer_dims = torch.tensor([3*64,64*64,64*128,128*128,128*256,
                                        256*256,256*256,256*512,512*512,
                                        512*512,512*512,512*512,512*512],
                                       requires_grad=False, dtype=torch.float32)
        self.Conv0 = child[0]   # (3,64)
        self.Btch1 = child[1]   # Batch
        self.Conv3 = child[3]   # (64,64)
        self.Btch4 = child[4]   # Batch
        self.Conv7 = child[7]   # (64,128)
        self.Btch8 = child[8]   # Batch
        self.Conv10 = child[10] # (128,128)
        self.Btch11 = child[11] # Batch
        self.Conv14 = child[14] # (128,256)
        self.Btch15 = child[15] # Batch
        self.Conv17 = child[17] # (256,256)
        self.Btch18 = child[18] # Batch
        self.Conv20 = child[20] # (256,256)
        self.Btch21 = child[21] # Batch
        self.Conv24 = child[24] # (256,512)
        self.Btch25 = child[25] # Batch
        self.Conv27 = child[27] # (512,512)
        self.Btch28 = child[28] # Batch
        self.Conv30 = child[30] # (512,512)
        self.Btch31 = child[31] # Batch
        self.Conv34 = child[34] # (512,512)
        self.Btch35 = child[35] # Batch
        self.Conv37 = child[37] # (512,512)
        self.Btch38 = child[38] # Batch
        self.Conv40 = child[40] # (512,512)
        self.Btch41 = child[41] # Batch
        layers = [self.Conv0, self.Conv3, self.Conv7, self.Conv10,
                  self.Conv14, self.Conv17, self.Conv20, self.Conv24,
                  self.Conv27, self.Conv30, self.Conv34, self.Conv37, self.Conv40]
        # constant in KL divergence
        self.w_squared = torch.tensor(0,requires_grad=False)
        self.layer_dimensions = torch.zeros(1, 13, requires_grad=True)
        for i, layer in enumerate(layers):
            params = next(layer.parameters())
            self.w_squared = self.w_squared + params.pow(2).sum()
            self.layer_dimensions[0,i] = params.size()

    def forward(self,x):
        out = self.reparameterize(self.Conv0(x), self.logvars[0,0])
        out = F.relu(self.Btch1(out))
        out = self.reparameterize(self.Conv3(out), self.logvars[0,1])
        out = F.max_pool2d(F.relu(self.Btch4(out)), 2, 2)
        out = self.reparameterize(self.Conv7(out), self.logvars[0,2])
        out = F.relu(self.Btch8(out))
        out = self.reparameterize(self.Conv10(out), self.logvars[0,3])
        out = F.max_pool2d(F.relu(self.Btch11(out)), 2, 2)
        out = self.reparameterize(self.Conv14(out), self.logvars[0,4])
        out = F.relu(self.Btch15(out))
        out = self.reparameterize(self.Conv17(out), self.logvars[0,5])
        out = F.relu(self.Btch18(out))
        out = self.reparameterize(self.Conv20(out), self.logvars[0,6])
        out = F.max_pool2d(F.relu(self.Btch21(out)), 2, 2)
        out = self.reparameterize(self.Conv24(out), self.logvars[0,7])
        out = F.relu(self.Btch25(out))
        out = self.reparameterize(self.Conv27(out), self.logvars[0,8])
        out = F.relu(self.Btch28(out))
        out = self.reparameterize(self.Conv30(out), self.logvars[0,9])
        out = F.max_pool2d(F.relu(self.Btch31(out)), 2, 2)
        out = self.reparameterize(self.Conv34(out), self.logvars[0,10])
        out = F.relu(self.Btch35(out))
        out = self.reparameterize(self.Conv37(out), self.logvars[0,11])
        out = F.relu(self.Btch38(out))
        out = self.reparameterize(self.Conv40(out), self.logvars[0,12])
        out = F.max_pool2d(F.relu(self.Btch41(out)), 2, 2)
        out = self.avgpool(out)
        out = torch.flatten(x, 1)
        out = self.classifier(out)
        return out

    def _load_pretrained_model(self):
        vgg_model = torchvision.models.vgg16_bn(pretrained=True)
        child_counter = 0
        for child in vgg_model.children():
            for param in child.parameters():
                param.requires_grad = False
        return vgg_model
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(mu)
        return mu + eps*std