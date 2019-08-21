import torch.nn as nn
import torch

class CBA(nn.Module):
    def __init__(self, in_channels = 16, out_channels = 16, kernel_size = 3, stride = 1, padding = 1, bias = True, groups = 1, with_sn = True, act=nn.LeakyReLU):
        super().__init__()
        if with_sn:
            self.model = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = bias, groups = groups)),
                nn.BatchNorm2d(out_channels),
                act()
            )
        else:
            self.model = nn.Sequential(
                nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = bias, groups = groups),
                nn.BatchNorm2d(out_channels),
                act()
            )

    def forward(self, x):
        return self.model(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels = 512, out_channels = 512):
        super().__init__()
        self.bypass = []
        self.model = []
        if in_channels != out_channels:
            self.bypass += [CBA(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)]
        self.model = nn.Sequential(
            CBA(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            CBA(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.bypass = nn.Sequential(*self.bypass)

    def forward(self, x):
        return self.bypass(x) + self.model(x)

class InferenceModel(nn.Module):
    def __init__(self, in_channels = 3, z_dim = 512):
        super().__init__()
        self.model = nn.Sequential(
            CBA(in_channels=in_channels, out_channels=16, kernel_size=5, stride=1, padding=2, with_sn=False),   # 64 x 64
            nn.AvgPool2d(kernel_size=2, stride=2),                                                              # 32 x 32
            ResBlock(in_channels=16, out_channels=32),                                                          # 32 x 32
            nn.AvgPool2d(kernel_size=2, stride=2),                                                              # 16 x 16
            ResBlock(in_channels=32, out_channels=64),                                                          # 16 x 16
            nn.AvgPool2d(kernel_size=2, stride=2),                                                              # 8 x 8
            ResBlock(in_channels=64, out_channels=128),                                                         # 8 x 8
            nn.AvgPool2d(kernel_size=2, stride=2),                                                              # 4 x 4
            ResBlock(in_channels=128, out_channels=128),                                                        # 4 x 4
        )
        self.mean = nn.Linear(z_dim * 16, z_dim)
        self.logvar = nn.Linear(z_dim * 16, z_dim)

    def forward(self, img):
        hidden = self.model(img)
        hidden = hidden.view(hidden.size(0), -1)
        mean = self.mean(hidden)
        logvar = self.logvar(hidden)
        return mean, logvar

class Generator(nn.Module):
    def __init__(self, z_dim = 512, out_channels = 3):
        super().__init__()
        self.fc = nn.Linear(z_dim, z_dim * 16)
        self.model = nn.Sequential(
            ResBlock(in_channels=128, out_channels=128),                                                        # 4 x 4
            nn.Upsample(scale_factor=2),                                                                        # 8 x 8
            ResBlock(in_channels=128, out_channels=64),                                                         # 8 x 8
            nn.Upsample(scale_factor=2),                                                                        # 16 x 16
            ResBlock(in_channels=64, out_channels=32),                                                          # 16 x 16
            nn.Upsample(scale_factor=2),                                                                        # 32 x 32
            ResBlock(in_channels=32, out_channels=16),                                                          # 32 x 32
            nn.Upsample(scale_factor=2),                                                                        # 64 x 64
            ResBlock(in_channels=16, out_channels=16),                                                          # 64 x 64
            CBA(in_channels=16, out_channels=out_channels, kernel_size=5, stride=1, padding=2, act=nn.Sigmoid),
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(z.size(0), -1, 4, 4)
        out = self.model(out)
        return out