from block import *


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN.
    W = (W - F + 2P) /S + 1"""

    def __init__(self, in_channel=1, spec_norm=True, LR=0.02):
        super(Discriminator, self).__init__()
        self.base = list()
        self.dis = list()
        self.cls = list()
        self.cls_fc = nn.Linear(128, 2)
        self.base.append(ConvBlock(in_channel, 16, spec_norm, stride=2, LR=LR)) # 256 -> 128
        self.base.append(ConvBlock(16, 32, spec_norm, stride=2, LR=LR)) # 128 -> 64
        self.base.append(ConvBlock(32, 64, spec_norm, stride=2, LR=LR)) # 64 -> 32
        self.base.append(ConvBlock(64, 128, spec_norm, stride=2, LR=LR)) # 32 -> 16
        self.dis.append(nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1))
        self.cls.append(nn.AdaptiveAvgPool2d((1,1)))


        self.base = nn.Sequential(*self.base)
        self.dis = nn.Sequential(*self.dis)
        self.cls = nn.Sequential(*self.cls)

    def forward(self, x):
        fx = self.cls(self.base(x))
        fx = fx.view(-1, 128)
        return self.dis(self.base(x)), self.cls_fc(fx)

class ResBlockNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResBlockNet, self).__init__()
        self.main = list()
        self.main.append(ResBlock(in_channels, out_channels))
        self.main.append(ResBlock(in_channels, out_channels))
        self.main.append(ResBlock(in_channels, out_channels))
        self.main.append(ResBlock(in_channels, out_channels))
        self.main = nn.Sequential(*self.main)

    def forward(self, x):
        return self.main(x)

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
        self.down4 = ConvBlock(512, 1024 // factor, LR=LR, stride=2, spec_norm=spec_norm)
        self.res_block = ResBlockNet(512, 512)
        self.up1 = Up(1024, 512 // factor, bilinear, LR=LR, spec_norm=spec_norm)
        self.up2 = Up(512, 256 // factor, bilinear, LR=LR, spec_norm=spec_norm)
        self.up3 = Up(256, 128 // factor, bilinear, LR=LR, spec_norm=spec_norm)
        self.up4 = Up(128, 64, bilinear, LR=LR, spec_norm=spec_norm)
        self.outc = LastConv(64, out_channels)

        # Classification Part
        self.conv2 = list()
        self.conv2.append(ResBlock(128,128))
        self.conv2.append(ConvBlock(128,128, LR=LR, stride=2, spec_norm=spec_norm))
        self.conv2 = nn.Sequential(*self.conv2)

        self.conv3 = list()
        self.conv3.append(ResBlock(64, 64))
        self.conv3.append(ConvBlock(64, 64, LR=LR, stride=2, spec_norm=spec_norm))
        self.conv3.append(ResBlock(64, 64))
        self.conv3.append(ConvBlock(64, 64, LR=LR, stride=2, spec_norm=spec_norm))
        self.conv3 = nn.Sequential(*self.conv3)

        self.conv4 = list()
        self.conv4.append(ResBlock(64, 64))
        self.conv4.append(ConvBlock(64, 64, LR=LR, stride=2, spec_norm=spec_norm))
        self.conv4.append(ResBlock(64, 64))
        self.conv4.append(ConvBlock(64, 64, LR=LR, stride=2, spec_norm=spec_norm))
        self.conv4.append(ResBlock(64, 64))
        self.conv4.append(ConvBlock(64, 64, LR=LR, stride=2, spec_norm=spec_norm))
        self.conv4 = nn.Sequential(*self.conv4)
        self.fc = nn.Linear(2048, 2)
        self.down_sample = nn.AdaptiveAvgPool2d((2,2))

    def forward(self, x):

        # unet part
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.res_block(x5)
        x_up1 = self.up1(x5, x4)
        x_up2 = self.up2(x_up1, x3)
        x_up3 = self.up3(x_up2, x2)
        x_up4 = self.up4(x_up3, x1)
        image = self.outc(x_up4)

        #classification part
        x_up1 = self.down_sample(x_up1)
        x_conv2 = self.down_sample(self.conv2(x_up2))
        x_conv3 = self.down_sample(self.conv3(x_up3))
        x_conv4 = self.down_sample(self.conv4(x_up4))

        x_concat = torch.cat([x_up1, x_conv2, x_conv3, x_conv4],dim=1)
        x_concat = x_concat.view(x_concat.size()[0], -1)
        pred = self.fc(x_concat)
        return image, self.fc(x_concat)
