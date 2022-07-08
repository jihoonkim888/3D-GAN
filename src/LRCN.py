import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.distributions import Categorical


class LRCN(nn.Module):
    def __init__(self, in_channels=1, input_dim=32, output_dim=128, in_conv_channels=256, out_conv_channels=64, latent_size=200, dropout_prob=0.5):
        super().__init__()
        img_feat_size = input_dim
        input_size = input_dim

        # 3D-GAN (encoder of 3D-ED-GAN)
        in_conv1_channels = int(in_conv_channels / 8)
        in_conv2_channels = int(in_conv_channels / 4)
        in_conv3_channels = int(in_conv_channels / 2)
        self.in_conv_channels = in_conv_channels

        self.in_conv1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=in_conv1_channels,
                      kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(in_conv1_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.in_conv2 = nn.Sequential(
            nn.Conv3d(in_channels=in_conv1_channels, out_channels=in_conv2_channels,
                      kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(in_conv2_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.in_conv3 = nn.Sequential(
            nn.Conv3d(in_channels=in_conv2_channels, out_channels=in_conv3_channels,
                      kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(in_conv3_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # self.in_conv4 = nn.Sequential(
        #     nn.Conv3d(in_channels=in_conv3_channels, out_channels=in_conv_channels,
        #               kernel_size=5, stride=2, padding=1, bias=False),
        #     nn.BatchNorm3d(in_conv_channels),
        #     nn.LeakyReLU(0.2, inplace=True)
        # )

        # self.last_layer_size = (input_dim//8)**3 * in_conv_channels
        self.last_layer_size = 128
        self.fc1 = nn.Linear(self.last_layer_size, latent_size)

        # # LSTM
        # self.lstm = nn.LSTM(input_size=latent_size, batch_first=True)
        # self.linear1 = nn.Linear(img_feat_size, hidden_size)
        # self.lstm1 = nn.LSTM(lstm1_input_size, hidden_size, batch_first=True)
        # self.lstm2 = nn.LSTM(lstm2_input_size, hidden_size, batch_first=True)
        # self.linear2 = nn.Linear(hidden_size, vocab_size)
        # self.init_weights()

        # self.input_size = (input_size, vocab_size)
        # self.output_size = vocab_size
        # self.dropout_prob = dropout_prob

        # # 2D FC GAN

    # def init_weights(self):
    #     self.word_embed.weight.data.uniform_(-0.1, 0.1)
    #     self.linear1.weight.data.uniform_(-0.1, 0.1)
    #     self.linear1.bias.data.fill_(0)
    #     self.linear2.weight.data.uniform_(-0.1, 0.1)
    #     self.linear2.bias.data.fill_(0)

    def forward(self, x):
        # 3D-CNN
        x = self.in_conv1(x)
        x = self.in_conv2(x)
        x = self.in_conv3(x)
        # x = self.in_conv4(x)
        x = x.view(-1, self.last_layer_size)
        x = self.fc1(x)

        # LSTM

    #     if feat_func is None:
    #         def feat_func(x): return x

    #     embeddings = self.word_embed(captions)
    #     embeddings = F.dropout(
    #         embeddings, p=self.dropout_prob, training=self.training)

    #     if self.has_vision_model:
    #         image_features = self.vision_model(image_inputs)
    #     else:
    #         image_features = image_inputs
    #     image_features = self.linear1(image_features)
    #     image_features = F.relu(image_features)
    #     image_features = F.dropout(
    #         image_features, p=self.dropout_prob, training=self.training)
    #     image_features = feat_func(image_features)
    #     image_features = image_features.unsqueeze(1)
    #     image_features = image_features.expand(-1, embeddings.size(1), -1)

    #     packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
    #     hiddens, _ = self.lstm1(packed)
    #     unpacked_hiddens, new_lengths = pad_packed_sequence(
    #         hiddens, batch_first=True)
    #     unpacked_hiddens = torch.cat((image_features, unpacked_hiddens), 2)
    #     unpacked_hiddens = F.dropout(
    #         unpacked_hiddens, p=self.dropout_prob, training=self.training)
    #     packed_hiddens = pack_padded_sequence(unpacked_hiddens, lengths,
    #                                           batch_first=True)
    #     hiddens, _ = self.lstm2(packed_hiddens)

    #     hiddens = F.dropout(
    #         hiddens[0], p=self.dropout_prob, training=self.training)
    #     outputs = self.linear2(hiddens)
    #     return outputs

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
