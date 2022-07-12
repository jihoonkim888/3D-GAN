import torch
import torch.nn as nn
# import torch.nn.functional as F


class Upscaler(nn.Module):
    def __init__(self, in_channels=1, input_dim=64, output_dim=128, conv_channels=256, batch_size=5):
        super().__init__()
        self.in_channels = in_channels
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scale = output_dim / input_dim
        self.batch_size = batch_size
        self.conv_channels = conv_channels
        self.conv1_channels = int(conv_channels / 4)
        self.conv2_channels = int(conv_channels / 2)
        self.in_conv1 = nn.Conv3d(in_channels=self.in_channels,
                                  out_channels=self.conv1_channels, kernel_size=4, stride=2, padding=1)
        self.in_conv2 = nn.Conv3d(in_channels=self.conv1_channels,
                                  out_channels=self.conv2_channels, kernel_size=4, stride=2, padding=1)
        self.in_conv3 = nn.Conv3d(in_channels=self.conv2_channels,
                                  out_channels=self.conv_channels, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(
            int(batch_size*conv_channels*(input_dim/8)**3), 200)
        self.fc2 = nn.Linear(
            200, int(batch_size*conv_channels*(input_dim/4)**3))
        self.out_conv1 = nn.ConvTranspose3d(in_channels=self.conv_channels,
                                            out_channels=self.conv2_channels, kernel_size=4, stride=2, padding=1)
        self.out_conv2 = nn.ConvTranspose3d(in_channels=self.conv2_channels,
                                            out_channels=self.conv1_channels, kernel_size=4, stride=2, padding=1)
        self.out_conv3 = nn.ConvTranspose3d(in_channels=self.conv1_channels,
                                            out_channels=self.in_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        linear_dim = int(self.input_dim / 16)
        x = self.in_conv1(x)
        x = self.in_conv2(x)
        x = self.in_conv3(x)
        x = x.view(1, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, self.conv_channels, linear_dim, linear_dim, linear_dim)
        x = self.out_conv1(x)
        x = self.out_conv2(x)
        x = self.out_conv3(x)
        return x
