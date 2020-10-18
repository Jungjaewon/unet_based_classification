from block import *


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN.
    W = (W - F + 2P) /S + 1"""

    def __init__(self, in_channel=3, spec_norm=True, LR=0.02):
        super(Discriminator, self).__init__()
        self.main = list()
        self.main.append(ConvBlock(in_channel, 16, spec_norm, stride=2, LR=LR)) # 256 -> 128
        self.main.append(ConvBlock(16, 32, spec_norm, stride=2, LR=LR)) # 128 -> 64
        self.main.append(ConvBlock(32, 64, spec_norm, stride=2, LR=LR)) # 64 -> 32
        self.main.append(ConvBlock(64, 128, spec_norm, stride=2, LR=LR)) # 32 -> 16
        self.main.append(nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1))
        self.main = nn.Sequential(*self.main)

    def forward(self, x):
        return self.main(x)

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
        self.main.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1))
        self.main.append(nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True))
        self.main.append(nn.ReLU(inplace=True))
        self.main.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1))
        self.main.append(nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True))
        self.main.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*self.main)

    def forward(self, x):
        return self.main(x) + x

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, bilinear=True, LR=0.02, spec_norm=False):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        # Unet part
        self.inc = ConvBlock(in_channels, 64, LR=LR, spec_norm=spec_norm)
        self.down1 = ConvBlock(64, 128, LR=LR, stride=2, spec_norm=spec_norm)
        self.down2 = ConvBlock(128, 256, LR=LR, stride=2, spec_norm=spec_norm)
        self.down3 = ConvBlock(256, 512, LR=LR, stride=2, spec_norm=spec_norm)
        factor = 2 if bilinear else 1
        self.res_block = ResBlockNet(512,512)
        self.down4 = ConvBlock(512, 1024 // factor, LR=LR, stride=2, spec_norm=spec_norm)
        self.up1 = Up(1024, 512 // factor, bilinear, LR=LR, spec_norm=spec_norm)
        self.up2 = Up(512, 256 // factor, bilinear, LR=LR, spec_norm=spec_norm)
        self.up3 = Up(256, 128 // factor, bilinear, LR=LR, spec_norm=spec_norm)
        self.up4 = Up(128, 64, bilinear, LR=LR, spec_norm=spec_norm)
        self.outc = LastConv(64, out_channels)

        # Classification Part
        self.conv1 = None
        self.conv2 = None
        self.conv3 = None
        self.conv4 = None

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.res_block(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
