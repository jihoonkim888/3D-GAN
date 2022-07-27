import os
import numpy as np
import torch
from src.GAN import Generator
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('-dp', '--data_path', type=str, required=True)
parser.add_argument('-wp', '--weight_path', type=str, required=True)
parser.add_argument('-we', '--weight_epoch', type=int, required=False)
parser.add_argument('-sp', '--save_path', type=str, required=True)
parser.add_argument('-n', '--n_samples', type=int, required=True)
parser.add_argument('-r', '--resolution', type=int, required=True)
args = parser.parse_args()

# argparse
# data_path = args.data_path
weights_path = args.weight_path
weight_epoch = args.weight_epoch if args.weight_epoch else None
save_path = args.save_path
n_samples = args.n_samples
dim = args.resolution
noise_dim = 200
b_size = 50
conv_channels = 256

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)


def init_netG():
    netG = Generator(in_channels=conv_channels, out_dim=dim,
                     out_channels=1, noise_dim=noise_dim)
    netG = netG.to(device)
    return netG


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
    netG = init_netG()
    epoch = find_weight_epoch()
    netG_filename = f'{weights_path}/netG_r{dim}_{epoch}.pth'
    print('weights to load:', netG_filename)
    netG.load_state_dict(torch.load(netG_filename))
    print('weights loaded')
    noises = torch.randn(n_samples, noise_dim)
    lst_samples = []
    for noise in torch.split(noises, b_size):
        with torch.no_grad():
            samples = netG(noise.to(device)).cpu()
            lst_samples.append(samples)
    output = torch.cat(lst_samples)
    output = output.numpy()
    print('Output shape:', output.shape)
    with open(save_path, 'wb') as f:
        np.save(f, output)
    print('Output saved at', save_path)
