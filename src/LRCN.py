from dataclasses import replace
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.distributions import Categorical


class LRCN(nn.Module):
    def __init__(self, in_channels=1, input_dim=64, output_dim=128, c=5, in_conv_channels=256, out_conv_channels=256, kernel_size=3, latent_size=200, hidden_size=100):
        super().__init__()
        self.in_channels = in_channels
        self.out_conv_channels = out_conv_channels
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.c = c

        # 3D-GAN (encoder of 3D-ED-GAN)
        in_conv1_channels = int(in_conv_channels / 4)
        in_conv2_channels = int(in_conv_channels / 2)
        in_conv3_channels = in_conv_channels
        self.in_conv_channels = in_conv_channels

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

        # self.last_layer_size = (input_dim//8)**3 * in_conv_channels
        # self.last_layer_size = 256 * 8 * 8
        self.fc1 = nn.Linear(in_features=in_conv_channels *
                             input_dim, out_features=latent_size)

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
            hidden_size, out_conv_channels * in_dim * in_dim)

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

        self.out_conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_conv1_channels,
                               out_channels=1,
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
        hidden = None
        lst_x = []
        # iterate through input model
        for t in range(x_input.size(1)):
            x = torch.reshape(x_input[:, t, :, :, :],
                              (-1, 1, self.input_dim, self.input_dim, 5))
            # 3D-CNN
            x = self.in_conv1(x)
            x = self.in_conv2(x)
            x = self.in_conv3(x)
            x = x.view(1, -1)
            x = self.fc1(x)

            # LSTM
            x, hidden = self.lstm(x, hidden)

            # 2D-CNN
            x = self.linear(x)
            x = x.view(-1, self.out_conv_channels,
                       self.in_dim, self.in_dim)
            x = self.out_conv1(x)
            x = self.out_conv2(x)

            lst_x.append(x)

        # x_ret = torch.stack(lst_x)
        # return x_ret

    # def pca(self):
    #     '''
    #     Principal component analysis module, required to find the best orientation of an input model before going through the network.
    #     '''
    #     return

    # def generate_sentence(self, image_inputs, start_word, end_word, states=(None, None),
    #                       max_sampling_length=50, sample=False, feat_func=None):
    #     if feat_func is None:
    #         def feat_func(x): return x

    #     sampled_ids = []
    #     if self.has_vision_model:
    #         image_features = self.vision_model(image_inputs)
    #     else:
    #         image_features = image_inputs
    #     image_features = self.linear1(image_features)
    #     image_features = F.relu(image_features)
    #     image_features = feat_func(image_features)
    #     image_features = image_features.unsqueeze(1)

    #     embedded_word = self.word_embed(start_word)
    #     embedded_word = embedded_word.expand(image_features.size(0), -1, -1)

    #     lstm1_states, lstm2_states = states

    #     end_word = end_word.squeeze().expand(image_features.size(0))
    #     reached_end = torch.zeros_like(end_word.data).byte()

    #     if sample:
    #         log_probabilities = []
    #         lengths = torch.zeros_like(reached_end).long()

    #     i = 0
    #     while not reached_end.all() and i < max_sampling_length:
    #         lstm1_input = embedded_word

    #         # LSTM 1
    #         lstm1_output, lstm1_states = self.lstm1(lstm1_input, lstm1_states)

    #         lstm1_output = torch.cat((image_features, lstm1_output), 2)

    #         # LSTM 2
    #         lstm2_output, lstm2_states = self.lstm2(lstm1_output, lstm2_states)

    #         outputs = self.linear2(lstm2_output.squeeze(1))
    #         if sample:
    #             predicted, log_p = self.sample(outputs)
    #             active_batches = (~reached_end)
    #             log_p *= active_batches.float().to(log_p.device)
    #             log_probabilities.append(log_p.unsqueeze(1))
    #             lengths += active_batches.long()
    #         else:
    #             predicted = outputs.max(1)[1]
    #         reached_end = reached_end | predicted.eq(end_word).data
    #         sampled_ids.append(predicted.unsqueeze(1))
    #         embedded_word = self.word_embed(predicted)
    #         embedded_word = embedded_word.unsqueeze(1)

    #         i += 1

    #     sampled_ids = torch.cat(sampled_ids, 1).squeeze()
    #     if sample:
    #         log_probabilities = torch.cat(log_probabilities, 1).squeeze()
    #         return sampled_ids, log_probabilities, lengths
    #     return sampled_ids


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
