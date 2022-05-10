import torch
import torch.nn as nn
from pix2pix.layers import DownBlock, UpBlock
# Encoder:
# C64-C128-C256-C512-C512-C512-C512-C512
# Decoder:
# CD512-CD512-CD512-C512-C256-C128-C64
class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down1 = DownBlock(64, use_norm=False)
        self.down2 = DownBlock(128)
        self.down3 = DownBlock(256)
        self.down4 = DownBlock(512)
        self.down5 = DownBlock(512)
        self.down6 = DownBlock(512)
        self.down7 = DownBlock(512)
        self.down8 = DownBlock(512, use_norm=False)

        self.up1 = UpBlock(512, True)
        self.up2 = UpBlock(512, True)
        self.up3 = UpBlock(512, True)
        self.up4 = UpBlock(512)
        self.up5 = UpBlock(256)
        self.up6 = UpBlock(128)
        self.up7 = UpBlock(64)
        self.up8 = nn.LazyConvTranspose2d(3, 4, 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        u7 = self.up7(torch.cat([u6, d2], dim=1))
        u8 = self.up8(torch.cat([u7, d1], dim=1))
        return self.tanh(u8)

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, DownBlock):
                nn.init.normal_(m.conv.weight, mean, std)
            if isinstance(m, UpBlock):
                nn.init.normal_(m.conv.weight, mean, std)

if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    model = Generator()
    model(x)
    model.normal_weight_init()
    print(model(x).shape)
