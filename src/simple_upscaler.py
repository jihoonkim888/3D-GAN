import torch
import torch.nn as nn
import torch.nn.functional as F


class Upscaler(nn.Module):
    def __init__(self, in_channels=1, input_dim=64, output_dim=256):
        '''
        Replicating the generator structure of Yang et al. (2018), available at: https://github.com/Yang7879/3D-RecGAN-extended
        '''
        super().__init__()
        self.in_channels = in_channels
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scale = output_dim / input_dim

        self.up_conv1_channels = 64
        self.up_conv2_channels = in_channels

        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=in_channels,
                               out_channels=self.up_conv1_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.up_conv1_channels,
                               out_channels=in_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.up_conv1(x)
        x = self.up_conv2(x)
        return x
