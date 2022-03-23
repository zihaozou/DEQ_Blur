import torch
from torch import nn


class DnCNN(nn.Module):
    def __init__(self, depth=12, in_channels=3, out_channels=3, init_features=64, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        layers = []

        layers.append(nn.Conv2d(
            in_channels=in_channels, out_channels=init_features, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ELU())

        for _ in range(depth-1):
            layers.append((nn.Conv2d(
                in_channels=init_features, out_channels=init_features, kernel_size=kernel_size, padding=padding, bias=True)))
            layers.append(nn.ELU())

        layers.append(nn.Conv2d(
            in_channels=init_features, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=True))

        self.dncnn_3 = nn.Sequential(*layers)

    def forward(self, x):
        return self.dncnn_3(x).mean([1,2,3]).sum()
