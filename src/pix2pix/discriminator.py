import torch
import torch.nn as nn
from pix2pix.layers import DownBlock

# C64-C128-C256-C512
class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            DownBlock(out_channels=64, use_norm=False),
            DownBlock(out_channels=128),
            DownBlock(256),
            DownBlock(512, last_layer=True),
            nn.LazyConv2d(1, kernel_size=4, padding=1, padding_mode="zeros")
        )

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        return self.model(x)

    def __weights_init(self, m, mean=0.0, std=0.02):
        if isinstance(m, DownBlock):
            torch.nn.init.normal_(m.conv.weight, mean, std)

    def normal_weight_init(self):
        self.model.apply(self.__weights_init)


if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator()
    model(x, y)
    model.normal_weight_init()
    print(model(x, y).shape)
