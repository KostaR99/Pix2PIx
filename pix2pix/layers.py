import torch.nn as nn


class DownBlock(nn.Module):
    def __init__(self, out_channels: int, use_norm: bool = True, last_layer: bool = False) -> None:
        super().__init__()

        self.conv = nn.LazyConv2d(
            out_channels=out_channels,
            kernel_size=4,
            stride=2 if not last_layer else 1,
            padding=1,
            bias=False if use_norm else True)
        self.use_norm = use_norm
        self.norm = nn.LazyInstanceNorm2d()
        self.leaky = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.conv(x)
        return self.leaky(self.norm(x)) if self.use_norm else self.leaky(x)


class UpBlock(nn.Module):
    def __init__(self, out_channels: int, use_dropout: bool = False) -> None:
        super().__init__()
        self.conv = nn.LazyConvTranspose2d(
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1)

        self.use_dropout = use_dropout
        self.drop = nn.Dropout(0.5)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        return self.relu(self.drop(x)) if self.use_dropout else self.relu(x)
