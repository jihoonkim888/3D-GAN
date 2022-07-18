import matplotlib.pyplot as plt
from src.upscaler import Upscaler, BCELoss_w
import argparse
from src import binvox_rw
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import random
from tqdm.auto import tqdm
import numpy as np
import os
# from torchinfo import summary


parser = argparse.ArgumentParser()
parser.add_argument('-id', '--input_dim', type=int, required=False)
parser.add_argument('-od', '--output_dim', type=int, required=False)
parser.add_argument('-dp', '--data_path', type=str,
                    required=True, help='Path to .npy file')
parser.add_argument('-wp', '--weight_path', type=str, required=True)
parser.add_argument('-e', '--epoch', type=int, required=False)
args = parser.parse_args()

# argparse
data_path = args.data_path
weights_path = args.weight_path
input_dim = args.input_dim if args.input_dim else 64
output_dim = args.output_dim if args.output_dim else 256
epoch = args.epoch if args.epoch else None

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)


def init_upscaler(input_dim, output_dim):
    net = Upscaler(input_dim=input_dim, output_dim=output_dim)
    return net


if __name__ == '__main__':
    net = init_upscaler(
        input_dim=input_dim, output_dim=output_dim)
    net = net.to(device)

    if epoch is None:
        weights_available = [i.strip('.pth').split('_')[-1]
                             for i in os.listdir(weights_path)]
        weights_available.sort()
        epoch = weights_available[-1]

    net_filename = f'net_r{input_dim}_r{output_dim}_{epoch}.pth'
    print('weights to load:', net_filename)
    net.load_state_dict(torch.load(
        os.path.join(weights_path, net_filename)))
    print('loaded weights on net with', net_filename)
    with open(data_path, 'rb') as f:
        arr = np.load(f)
    input_tensors = torch.tensor(arr)
    with torch.no_grad():
        output = net(input_tensors.to(device))
        output = output.cpu().numpy()
    with open('upscaled.npy', 'wb') as f:
        np.save(f, output)
    print('saved output as upscaled.npy')
