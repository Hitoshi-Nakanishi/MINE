import torch
from torch import nn

class ReparamterNorm(nn.Module):
    def __init__(self, input_dims, hidden_dim, reparameterization = True):
        """
        If reparameterization = False, it returns just x
        
        input_dims ... (C, H, W)
        """
        super().__init__()
        self.reparameterization = reparameterization
        self.logvar = nn.Sequential(
          nn.Conv2d(input_dims[0], hidden_dim, kernel_size=3, padding=1),
          nn.ReLU(), nn.Dropout(),
          nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
          nn.ReLU(), nn.Dropout(),
          nn.Conv2d(hidden_dim, input_dims[0], kernel_size=3, padding=1),
          nn.ReLU(),
          nn.AvgPool2d((input_dims[1], input_dims[2]))
        )

    def forward(self, x):
        repara_pairs = self.reparameterize(x)
        return repara_pairs
    
    def reparameterize(self, mu):
        std = torch.exp(0.5*self.logvar(mu))
        if self.reparameterization:
            eps = torch.randn_like(mu)
            return mu + 0.1 * eps * std, std
        else:
            return mu, std

class MultiSequential(nn.Sequential):
    def __init__(self, contexts = None, *args):
        """
        """
        super().__init__(*args)
        if contexts is not None:
            assert len(contexts) == len(args), "contexts should have same dimension with args"
        self.contexts = contexts

    def forward(self, input):
        if self.contexts is None:
            for module in self._modules.values():
                input = module(input)
            return input
        else:
            multi_outputs = []
            for ctx, module in zip(self.contexts, self._modules.values()):
                if ctx == 1:
                    input, output = module(input)
                    multi_outputs.append(output)
                else:
                    input = module(input)                    
            return input, multi_outputs    