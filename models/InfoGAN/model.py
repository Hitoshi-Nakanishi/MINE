import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_size=74, output_size=1, hidden_size=None):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(input_size, 1024, 1, 1, bias=False),
            nn.BatchNorm2d(1024), nn.ReLU(),
            nn.ConvTranspose2d(1024, 128, 7, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, output_size, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1, 0.02)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_size=None):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_size, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.1),
            nn.Conv2d(128, 1024, 7, bias=False),
            nn.BatchNorm2d(1024), nn.LeakyReLU(0.1),
            nn.Conv2d(1024, output_size, 1, bias=False)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.utils.spectral_norm(m, n_power_iterations=2)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1, 0.02)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        return self.main(x)

class ContStatistics(nn.Module):
    def __init__(self, input_size=28**2 + 1, output_size=1, hidden_size=1024):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size), nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size), nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, output_size, bias=False)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, 1, 0.02)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        return self.main(x)

class DiscStatistics(nn.Module):
    def __init__(self, input_size=28**2 + 10, output_size=1, hidden_size=1024):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size), nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size), nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, output_size, bias=False)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, 1, 0.02)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        return self.main(x)