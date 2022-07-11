import torch
import torch.nn as nn
# import torch.nn.functional as F


class Upscaler(nn.Module):
    def __init__(self, in_channels=1, input_dim=64, output_dim=128):
        super().__init__()
        self.in_channels = in_channels
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scale = output_dim / input_dim
        self.ups = nn.Upsample(scale_factor=self.scale)

    def forward(self, x):
        x = self.ups(x)
        return x
