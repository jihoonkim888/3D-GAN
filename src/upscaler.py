import torch
import torch.nn as nn
import torch.nn.functional as F


class Upscaler(nn.Module):
    def __init__(self, batch_size, in_channels=1, input_dim=64, output_dim=256, e_conv_channels=512, d_conv_channels=512, latent_size=2000):
        '''
        Replicating the generator structure of Yang et al. (2018), available at: https://github.com/Yang7879/3D-RecGAN-extended
        '''
        super().__init__()
        self.in_channels = in_channels
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scale = output_dim / input_dim
        self.latent_size = latent_size
        self.batch_size = batch_size

        self.e_conv1_channels = in_channels  # 1
        self.e_conv2_channels = int(e_conv_channels / 8)  # 64
        self.e_conv3_channels = int(e_conv_channels / 4)  # 128
        self.e_conv4_channels = int(e_conv_channels / 2)  # 256
        self.e_conv_channels = e_conv_channels  # 512

        self.e_conv1 = nn.Sequential(
            nn.Conv3d(in_channels=self.in_channels,
                      out_channels=self.e_conv2_channels, kernel_size=4, stride=1),
            F.pad()
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.LeakyReLU(),
        )

        self.e_conv2 = nn.Sequential(
            nn.Conv3d(in_channels=self.e_conv2_channels,
                      out_channels=self.e_conv3_channels, kernel_size=4, stride=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.LeakyReLU(),
        )

        self.e_conv3 = nn.Sequential(
            nn.Conv3d(in_channels=self.e_conv3_channels,
                      out_channels=self.e_conv4_channels, kernel_size=4, stride=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.LeakyReLU(),
        )

        self.e_conv4 = nn.Sequential(
            nn.Conv3d(in_channels=self.e_conv4_channels,
                      out_channels=self.e_conv_channels, kernel_size=4, stride=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.LeakyReLU(),
        )

        fc1_in_channels = int(((self.input_dim/16)**3)*self.e_conv_channels)
        fc2_out_channels = fc1_in_channels

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=fc1_in_channels,
                      out_features=self.latent_size)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=self.latent_size,
                      out_features=fc2_out_channels),
        )

        self.d_conv_channels = d_conv_channels  # 512
        self.d_conv2_channels = int(d_conv_channels / 2)  # 256
        self.d_conv3_channels = int(d_conv_channels / 4)  # 128
        self.d_conv4_channels = int(d_conv_channels / 8)  # 64
        self.d_conv5_channels = int(d_conv_channels / 32)  # 16

        self.d_conv1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.d_conv_channels,
                               out_channels=self.d_conv2_channels, kernel_size=4),
            nn.ReLU(),
        )  # 512 to 256

        self.d_conv2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.d_conv2_channels,
                               out_channels=self.d_conv3_channels, kernel_size=4),
            nn.ReLU(),
        )  # 256 to 128

        self.d_conv3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.d_conv3_channels,
                               out_channels=self.d_conv4_channels, kernel_size=4),
            nn.ReLU(),
        )  # 128 to 64

        self.d_conv4 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.d_conv4_channels,
                               out_channels=self.d_conv5_channels, kernel_size=4),
            nn.ReLU(),
        )  # 64 to 16

        self.up_conv1_channels = int(d_conv_channels / 64)
        self.up_conv2_channels = in_channels

        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.d_conv5_channels,
                               out_channels=self.up_conv1_channels, kernel_size=4),
            nn.ReLU(),
        )  # 16 to 8

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.up_conv1_channels,
                               out_channels=self.up_conv2_channels, kernel_size=4),
            nn.ReLU(),
        )  # 8 to 1

    def forward(self, x):
        post_dim = self.input_dim/16
        x = self.e_conv1(x)
        x = self.e_conv2(x)
        x = self.e_conv3(x)
        x = self.e_conv4(x)
        x = x.view(self.batch_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(self.batch_size, self.conv_channels,
                   post_dim, post_dim, post_dim)
        x = self.d_conv1(x)
        x = self.d_conv2(x)
        x = self.d_conv3(x)
        x = self.d_conv4(x)
        x = self.up_conv1(x)
        x = self.up_conv2(x)
        return x
