import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, stride=1, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
        )

    def forward(self, x, y):
        # x: [B,C,H,W], y: [B,1,H,W]
        if x.shape[-2:] != y.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="nearest")
        inp = torch.cat([x, y], dim=1)
        return self.net(inp)

class SpectralNormDiscriminator(nn.Module):
    def __init__(self, in_channels=13):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 64, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(256, 512, 4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(512, 1, 4, stride=1, padding=1)),
        )

    def forward(self, x, y):
        if x.shape[-2:] != y.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="nearest")
        return self.net(torch.cat([x, y], dim=1))
