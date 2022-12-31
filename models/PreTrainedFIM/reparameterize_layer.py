import torch
from torch import nn

class ReparamterNorm(nn.Module):
    def __init__(self, reparameterization = True):
        """
        If reparameterization = False, it returns just x.
        """
        super().__init__()
        self.reparameterization = reparameterization
        self.logvar = torch.tensor(0.1, requires_grad=True)

    def forward(self, x):
        repara_x = self.reparameterize(x)
        return repara_x
    
    def reparameterize(self, mu):
        if self.reparameterization:
            std = torch.exp(0.5*self.logvar)
            eps = torch.randn_like(mu)
            return mu + eps * std
        else:
            return mu