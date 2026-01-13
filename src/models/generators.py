import torch
import torch.nn as nn

from .blocks import ChebyKANConv2d, ChebyKANLayer, AttentionBlock, ResidualBlock

class UNetChebyKANGenerator(nn.Module):
    def __init__(self, in_channels=12, out_channels=1, kan_location="encoder"):
        super().__init__()
        self.kan_loc = kan_location
        self.pool = nn.MaxPool2d(2)

        if self.kan_loc in ["encoder", "both", "encoder_bottleneck"]:
            self.e1 = self.cb_with_kan(in_channels, 64)
            self.e2 = self.cb_with_kan(64, 128)
            self.e3 = self.cb_with_kan(128, 256)
            self.e4 = self.cb_with_kan(256, 512)
        else:
            self.e1 = self.cb(in_channels, 64)
            self.e2 = self.cb(64, 128)
            self.e3 = self.cb(128, 256)
            self.e4 = self.cb(256, 512)

        if self.kan_loc in ["bottleneck", "encoder_bottleneck", "bottleneck_decoder"]:
            self.bot_conv = nn.Conv2d(512, 512, 3, padding=1)
            self.ck_bot = ChebyKANLayer(512, 512, degree=4)
            self.ln_bot = nn.LayerNorm(512)
            up_in = 512
        else:
            self.bot = self.cb(512, 1024)
            up_in = 1024

        if self.kan_loc in ["decoder", "both", "bottleneck_decoder"]:
            self.u4 = self.ub_with_kan(up_in, 512)
            self.u3 = self.ub_with_kan(1024, 256)
            self.u2 = self.ub_with_kan(512, 128)
            self.u1 = self.ub_with_kan(256, 64)
        else:
            self.u4 = self.ub(up_in, 512)
            self.u3 = self.ub(1024, 256)
            self.u2 = self.ub(512, 128)
            self.u1 = self.ub(256, 64)

        self.final = nn.Sequential(nn.Conv2d(128, out_channels, 1), nn.Sigmoid())

    def cb(self, i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
        )

    def cb_with_kan(self, i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
            ChebyKANConv2d(o),
        )

    def ub(self, i, o):
        return nn.Sequential(nn.ConvTranspose2d(i, o, 2, stride=2), nn.BatchNorm2d(o), nn.ReLU(True))

    def ub_with_kan(self, i, o):
        return nn.Sequential(
            nn.ConvTranspose2d(i, o, 2, stride=2), nn.BatchNorm2d(o), nn.ReLU(True),
            ChebyKANConv2d(o),
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))

        if self.kan_loc in ["bottleneck", "encoder_bottleneck", "bottleneck_decoder"]:
            b = self.pool(e4)
            b = self.bot_conv(b)
            B, C, H, W = b.shape
            flat = b.permute(0, 2, 3, 1).reshape(-1, C)
            flat = self.ck_bot(flat)
            flat = self.ln_bot(flat)
            b = flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            b = self.bot(self.pool(e4))

        d4 = self.u4(b)
        d4 = torch.cat([d4, e4], 1)
        d3 = self.u3(d4)
        d3 = torch.cat([d3, e3], 1)
        d2 = self.u2(d3)
        d2 = torch.cat([d2, e2], 1)
        d1 = self.u1(d2)
        d1 = torch.cat([d1, e1], 1)
        return self.final(d1)

class Pix2PixGenerator(nn.Module):
    def __init__(self, in_channels=12, out_channels=1):
        super().__init__()
        self.enc1 = self._down(in_channels, 64, use_bn=False)
        self.enc2 = self._down(64, 128)
        self.enc3 = self._down(128, 256)
        self.enc4 = self._down(256, 512)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True)
        )
        self.up4 = self._up(512, 256)
        self.up3 = self._up(512, 128)
        self.up2 = self._up(256, 64)
        self.up1 = self._up(128, 64)
        self.final = nn.Sequential(nn.Conv2d(64, out_channels, 3, padding=1), nn.Sigmoid())

    def _down(self, in_ch, out_ch, use_bn=True):
        layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, True))
        return nn.Sequential(*layers)

    def _up(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)
        d4 = self.up4(b); d4 = torch.cat([d4, e3], 1)
        d3 = self.up3(d4); d3 = torch.cat([d3, e2], 1)
        d2 = self.up2(d3); d2 = torch.cat([d2, e1], 1)
        d1 = self.up1(d2)
        return self.final(d1)

class CycleGANGenerator(nn.Module):
    def __init__(self, in_channels=12, out_channels=1, num_res=6):
        super().__init__()
        self.init = nn.Sequential(nn.Conv2d(in_channels, 64, 7, padding=3), nn.BatchNorm2d(64), nn.ReLU(True))
        self.down1 = self.db(64, 128)
        self.down2 = self.db(128, 256)
        self.res = nn.Sequential(*[ResidualBlock(256) for _ in range(num_res)])
        self.up1 = self.ub(256, 128)
        self.up2 = self.ub(128, 64)
        self.final = nn.Sequential(nn.Conv2d(64, out_channels, 7, padding=3), nn.Sigmoid())

    def db(self, i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, stride=2, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))

    def ub(self, i, o):
        return nn.Sequential(nn.ConvTranspose2d(i, o, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(o), nn.ReLU(True))

    def forward(self, x):
        x = self.init(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.res(x)
        x = self.up1(x)
        x = self.up2(x)
        return self.final(x)

class AttentionUNetGenerator(nn.Module):
    def __init__(self, in_channels=12, out_channels=1):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.c1 = self.cb(in_channels, 64)
        self.c2 = self.cb(64, 128)
        self.c3 = self.cb(128, 256)
        self.c4 = self.cb(256, 512)
        self.att4 = AttentionBlock(512)
        self.att3 = AttentionBlock(256)
        self.att2 = AttentionBlock(128)
        self.att1 = AttentionBlock(64)
        self.bot = self.cb(512, 1024)
        self.u4 = self.ub(1024, 512)
        self.u3 = self.ub(1024, 256)
        self.u2 = self.ub(512, 128)
        self.u1 = self.ub(256, 64)
        self.final = nn.Sequential(nn.Conv2d(128, out_channels, 1), nn.Sigmoid())

    def cb(self, i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
        )

    def ub(self, i, o):
        return nn.Sequential(nn.ConvTranspose2d(i, o, 2, stride=2), nn.BatchNorm2d(o), nn.ReLU(True))

    def forward(self, x):
        e1 = self.c1(x)
        e2 = self.c2(self.pool(e1))
        e3 = self.c3(self.pool(e2))
        e4 = self.c4(self.pool(e3))
        b = self.bot(self.pool(e4))
        d4 = self.u4(b); d4 = torch.cat([d4, self.att4(e4)], 1)
        d3 = self.u3(d4); d3 = torch.cat([d3, self.att3(e3)], 1)
        d2 = self.u2(d3); d2 = torch.cat([d2, self.att2(e2)], 1)
        d1 = self.u1(d2); d1 = torch.cat([d1, self.att1(e1)], 1)
        return self.final(d1)

class UNetPlusPlusGenerator(nn.Module):
    def __init__(self, in_channels=12, out_channels=1):
        super().__init__()
        nb = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.c00 = self.cb(in_channels, nb[0])
        self.c10 = self.cb(nb[0], nb[1])
        self.c20 = self.cb(nb[1], nb[2])
        self.c30 = self.cb(nb[2], nb[3])
        self.c40 = self.cb(nb[3], nb[4])

        self.c01 = self.cb(nb[0] + nb[1], nb[0])
        self.c11 = self.cb(nb[1] + nb[2], nb[1])
        self.c21 = self.cb(nb[2] + nb[3], nb[2])
        self.c31 = self.cb(nb[3] + nb[4], nb[3])

        self.c02 = self.cb(nb[0]*2 + nb[1], nb[0])
        self.c12 = self.cb(nb[1]*2 + nb[2], nb[1])
        self.c22 = self.cb(nb[2]*2 + nb[3], nb[2])

        self.c03 = self.cb(nb[0]*3 + nb[1], nb[0])
        self.c13 = self.cb(nb[1]*3 + nb[2], nb[1])

        self.c04 = self.cb(nb[0]*4 + nb[1], nb[0])
        self.final = nn.Sequential(nn.Conv2d(nb[0], out_channels, 1), nn.Sigmoid())

    def cb(self, i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
        )

    def forward(self, x):
        x00 = self.c00(x)
        x10 = self.c10(self.pool(x00))
        x01 = self.c01(torch.cat([x00, self.up(x10)], 1))

        x20 = self.c20(self.pool(x10))
        x11 = self.c11(torch.cat([x10, self.up(x20)], 1))
        x02 = self.c02(torch.cat([x00, x01, self.up(x11)], 1))

        x30 = self.c30(self.pool(x20))
        x21 = self.c21(torch.cat([x20, self.up(x30)], 1))
        x12 = self.c12(torch.cat([x10, x11, self.up(x21)], 1))
        x03 = self.c03(torch.cat([x00, x01, x02, self.up(x12)], 1))

        x40 = self.c40(self.pool(x30))
        x31 = self.c31(torch.cat([x30, self.up(x40)], 1))
        x22 = self.c22(torch.cat([x20, x21, self.up(x31)], 1))
        x13 = self.c13(torch.cat([x10, x11, x12, self.up(x22)], 1))
        x04 = self.c04(torch.cat([x00, x01, x02, x03, self.up(x13)], 1))

        return self.final(x04)
