import torch
import torch.nn as nn
import torch.nn.functional as F
from spectral_normalization import SpectralNorm


class ResBlockNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResBlockNet, self).__init__()
        self.main = list()
        self.main.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1))
        self.main.append(nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True))
        self.main.append(nn.ReLU(inplace=True))
        self.main.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1))
        self.main.append(nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True))
        self.main.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*self.main)

    def forward(self, x):
        return self.main(x) + x

class ConvBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, spec_norm=False, LR=0.02, stride=1):
        super().__init__()
        if spec_norm:
            self.main = nn.Sequential(
                SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
                nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True),
                nn.LeakyReLU(LR, inplace=False),
                SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)),
                nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True),
                nn.LeakyReLU(LR, inplace=False),
            )

        else:
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True),
                nn.LeakyReLU(LR, inplace=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True),
                nn.LeakyReLU(LR, inplace=False),
            )

    def forward(self, x):
        return self.main(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, LR=0.02, spec_norm=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ConvBlock(in_channels, out_channels, spec_norm, LR)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_channels, out_channels, spec_norm, LR)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class LastConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LastConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.than_h = nn.Tanh()

    def forward(self, x):
        return self.tan_h(self.conv(x))
