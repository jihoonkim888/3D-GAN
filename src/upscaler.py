import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchinfo import summary


class Upscaler(nn.Module):
    def __init__(self, in_channels=1, conv_channels=64, input_dim=64, output_dim=256):
        '''
        Replicating the generator structure of Yang et al. (2018), available at: https://github.com/Yang7879/3D-RecGAN-extended
        '''
        super().__init__()
        self.in_channels = in_channels
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scale = output_dim / input_dim

        self.up_conv1_channels = conv_channels
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


def BCELoss_w(output, target, weights=None, eps=1e-10):
    output = output + eps
    if weights is not None:
        assert len(weights) == 2
        loss = weights[1] * (target * torch.log(output)) + \
            weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = torch.mul(target, torch.log(output)) + \
            torch.mul((1 - target), torch.log(1 - output))
    return torch.neg(torch.mean(loss))


# if __name__ == '__main__':
#     input_dim = 64
#     output_dim = 128
#     conv_channels = 64

#     input_size = (1, 1, input_dim, input_dim, input_dim)
#     ups = Upscaler(input_dim=input_dim, output_dim=output_dim,
#                    conv_channels=conv_channels)
#     summary(ups, input_size)
