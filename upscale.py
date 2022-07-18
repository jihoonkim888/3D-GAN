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
parser.add_argument('-dp', '--data_path', type=str,
                    required=True, help='Path to .npy file')
parser.add_argument('-wp', '--weight_path', type=str, required=True)
parser.add_argument('-sp', '--save_path', type=str, required=True)
parser.add_argument('-we', '--weight_epoch', type=int, required=False)
args = parser.parse_args()

# argparse
data_path = args.data_path
weights_path = args.weight_path
save_path = args.save_path
input_dim = 64
output_dim = 256
weight_epoch = args.weight_epoch if args.weight_epoch else None
b_size = 50

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)


def init_upscaler(input_dim, output_dim):
    net = Upscaler(input_dim=input_dim, output_dim=output_dim)
    return net


def find_weight_epoch():
    if weight_epoch is not None:
        epoch = weight_epoch
    else:
        weights_available = [i.strip('.pth').split('_')[-1]
                             for i in os.listdir(weights_path)]
        weights_available.sort()
        epoch = weights_available[-1]
    return epoch


if __name__ == '__main__':
    net = init_upscaler(
        input_dim=input_dim, output_dim=output_dim)
    net = net.to(device)

    weight_epoch = find_weight_epoch()

    net_filename = f'net_r{input_dim}_r{output_dim}_{weight_epoch}.pth'
    print('weights to load:', net_filename)
    net.load_state_dict(torch.load(
        os.path.join(weights_path, net_filename)))
    print('loaded weights on net with', net_filename)
    with open(data_path, 'rb') as f:
        arr = np.load(f)
    input_tensors = torch.tensor(arr)
    lst_samples = []
    for t in torch.split(input_tensors, b_size):
        with torch.no_grad():
            samples = net(t.to(device)).cpu()
            lst_samples.append(samples)
    output = torch.cat(lst_samples)
    output = output.numpy()
    print('Output shape:', output.shape)
    with open(save_path, 'wb') as f:
        np.save(f, output)
    print('Output saved at', save_path)
