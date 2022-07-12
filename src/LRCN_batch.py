from dataclasses import replace
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.distributions import Categorical


class LRCN(nn.Module):
    def __init__(self, in_channels=1, input_dim=64, output_dim=128, c=5, in_conv_channels=256, out_conv_channels=256, kernel_size=3, latent_size=200, hidden_size=100, batch_size=10):
        super().__init__()
        self.in_channels = in_channels
        self.in_conv_channels = in_conv_channels
        self.out_conv_channels = out_conv_channels
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.c = c

        # 3D-GAN (encoder of 3D-ED-GAN)
        in_conv1_channels = int(in_conv_channels / 4)
        in_conv2_channels = int(in_conv_channels / 2)
        in_conv3_channels = int(in_conv_channels)

        self.in_conv1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=in_conv1_channels,
                      kernel_size=kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(in_conv1_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.in_conv2 = nn.Sequential(
            nn.Conv3d(in_channels=in_conv1_channels, out_channels=in_conv2_channels,
                      kernel_size=kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(in_conv2_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.in_conv3 = nn.Sequential(
            nn.Conv3d(in_channels=in_conv2_channels, out_channels=in_conv3_channels,
                      kernel_size=kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(in_conv3_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # self.in_conv4 = nn.Sequential(
        #     nn.Conv3d(in_channels=in_conv3_channels, out_channels=in_conv4_channels,
        #               kernel_size=kernel_size, stride=2, padding=1, bias=False),
        #     nn.BatchNorm3d(in_conv4_channels),
        #     nn.LeakyReLU(0.2, inplace=True)
        # )

        # self.last_layer_size = (input_dim//8)**3 * in_conv_channels
        # self.last_layer_size = 256 * 8 * 8
        self.fc1 = nn.Linear(in_features=int(
            in_conv_channels*input_dim*batch_size), out_features=latent_size)

        # LSTM
        self.lstm = nn.LSTM(input_size=latent_size,
                            hidden_size=hidden_size,
                            batch_first=True
                            )

        # 2D CNN, two fully convolutional layers of kernel size 5 and stride 2, with batch norm and relu in between followed by tanh at the end
        in_dim = int(output_dim / 4)
        self.in_dim = in_dim
        out_conv1_channels = int(out_conv_channels / 2)

        self.linear = torch.nn.Linear(
            hidden_size, batch_size * out_conv_channels * in_dim * in_dim)

        self.out_conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_conv_channels,
                               out_channels=out_conv1_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False,
                               ),
            nn.BatchNorm2d(out_conv1_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        scale = int(output_dim / input_dim)

        self.out_conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_conv1_channels,
                               out_channels=scale,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False,
                               ),
            nn.Tanh()
            # nn.BatchNorm2d(out_conv2_channels),
            # nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x_input):
        '''
        x_input: 5D tensor, size of (batch, channel, depth, height, width)
        c: number of slices to input for each LRCN
        '''
        hidden = None
        lst_x = []
        lst_slices = []

        # # slice per batch
        # for b in range(x_input.size(0)):
        #     x = x_input[b][0]  # 5D to 3D
        #     x = self.slice_model(x)
        #     lst_slices.append(x)

        for b in range(x_input.size(0)):
            x = x_input[b][0]  # 5D to 3D
            x = self.slice_model(x)
            x = torch.stack(x).reshape(self.input_dim, self.in_channels,
                                       self.input_dim, self.input_dim, self.c)  # lists of 3D to 5D tensor
            lst_slices.append(x.unsqueeze(0))

        arr = torch.cat(lst_slices)  # list of 5D to 6D tensor
        arr = torch.reshape(arr, (self.input_dim, arr.size()[
                            0], self.in_channels, self.input_dim, self.input_dim, self.c))

        # for slices in lst_slices:
        #     lst_x_slice = []
        #     for slice in slices:
        #         x = torch.reshape(
        #             slice, (-1, self.in_channels, self.input_dim, self.input_dim, self.c))
        #         # 3D-CNN
        #         x = self.in_conv1(x)
        #         x = self.in_conv2(x)
        #         x = self.in_conv3(x)
        #         # x = self.in_conv4(x)
        #         x = x.view(1, -1)
        #         x = self.fc1(x)

        #         # LSTM
        #         x, hidden = self.lstm(x, hidden)

        #         # 2D-CNN
        #         x = self.linear(x)
        #         x = x.view(-1, self.out_conv_channels,
        #                    self.in_dim, self.in_dim)
        #         x = self.out_conv1(x)
        #         x = self.out_conv2(x)
        #         lst_x_slice.append(x[0])

        #     model = torch.cat(lst_x_slice, dim=0)
        #     # print('model shape:', model.shape)
        #     lst_x.append(model)

        lst_x = []
        for i in range(arr.size(0)):
            x = arr[i]  # 5D tensor

            # 3D-CNN
            x = self.in_conv1(x)
            x = self.in_conv2(x)
            x = self.in_conv3(x)
            # x = self.in_conv4(x)
            x = x.view(1, -1)
            x = self.fc1(x)

            # LSTM
            x, hidden = self.lstm(x, hidden)

            # 2D-CNN
            x = self.linear(x)
            x = x.view(-1, self.out_conv_channels,
                       self.in_dim, self.in_dim)
            # print('x shape after lstm and linear:', x.size())
            x = self.out_conv1(x)
            x = self.out_conv2(x)
            # print('x shape after out conv2:', x.size())
            lst_x.append(x)

        # print('lstx shape:', lst_x[0].shape)
        x_ret = torch.stack(lst_x).reshape(-1, self.in_channels,
                                           self.output_dim, self.output_dim, self.output_dim)
        # x_ret = torch.reshape(x_ret, (x_input.size(
        #     0), self.in_channels, -1, self.output_dim, self.output_dim))
        # print('x_ret shape:', x_ret.shape)
        return x_ret

    # def pca(self):
    #     '''
    #     Principal component analysis module, required to find the best orientation of an input model before going through the network.
    #     '''
    #     return

    def slice_model(self, x_3d):
        '''
        input: torch tensor of complete 3D model
        output: a list of parsed tensors with c
        '''
        lst = []
        i_start_to_pad = int((self.c-1)/2)
        i_end_to_pad = x_3d.shape[0] - int((self.c-1)/2) - 1  # 32-2-1 = 29
        for i in range(x_3d.shape[0]):
            start = int(i-((self.c-1)/2))
            start = start if start > 0 else 0
            end = int(i+((self.c+1)/2))
            # print('start:', start, 'end:', end)
            # x_s = torch.narrow(x_3d, -1, start, c)   # c slices of model
            x_s = x_3d[:, :, start:end]
            if i < i_start_to_pad:
                num_pad = i_start_to_pad - i
                x_s = F.pad(x_s, (num_pad, 0, 0, 0, 0, 0),
                            mode='constant', value=0)
            elif i > i_end_to_pad:  # i larger than 29, so 30 and 31
                num_pad = i - i_end_to_pad
                x_s = F.pad(x_s, (0, num_pad, 0, 0, 0, 0),
                            mode='constant', value=0)
            # print(x_s.shape)
            lst.append(x_s)

        return lst


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def calculate_input_size_conv2d(output_size, padding, dilation, kernel_size, stride):
    # equation got from conv2d torch website
    return (output_size - 1) * stride + 1 + dilation * (kernel_size+1) - 2 * padding


def calculate_input_size_convtranspose2d(output_size, padding, dilation, kernel_size, stride, output_padding=0):
    # equation got from conv2d torch website
    return (output_size-1-output_padding-dilation*(kernel_size-1)+2*padding) / stride + 1

# x_3d = torch.randn((32, 32, 32))
# print(x_3d)
# slices = slice(x_3d, c=5)
# print(slices[0])


# print(calculate_input_size_convtranspose2d(output_size=128,
#       padding=1, dilation=1, kernel_size=5, stride=2))
# print(calculate_input_size_convtranspose2d(output_size=64,
#                                            padding=1, dilation=1, kernel_size=5, stride=2))


# input = torch.randn(3, 1, 64, 64, 64)
# net = LRCN(input_dim=64, batch_size=3)
# output = net(input)
# print(output.size())
