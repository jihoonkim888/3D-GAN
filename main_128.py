import os
import numpy as np
import torch
from src.GAN import Generator
import argparse
from src.upscaler import Upscaler
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--n_samples', type=int, required=True)
parser.add_argument('-gwp', '--gen_weight_path', type=str, required=True)
parser.add_argument('-gwe', '--gen_weight_epoch', type=int, required=False)
parser.add_argument('-sp', '--save_path', type=str, required=True)
parser.add_argument('-gb', '--gen_batch_size', type=int, required=False)
args = parser.parse_args()

# argparse
gen_weight_path = args.gen_weight_path
gen_weight_epoch = args.gen_weight_epoch if args.gen_weight_epoch else None
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)
n_samples = args.n_samples

# parameters
dim = 128
noise_dim = 200
gen_b_size = args.gen_batch_size if args.gen_batch_size else 20
conv_channels = 256

elev = 30
azim = 135

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)


def init_netG():
    netG = Generator(in_channels=conv_channels, out_dim=dim,
                     out_channels=1, noise_dim=noise_dim)
    netG = netG.to(device)
    return netG


def find_weight_epoch(weight_path, weight_epoch):
    if weight_epoch is not None:
        epoch = weight_epoch
    else:
        weights_available = [int(i.strip('.pth').split('_')[-1])
                             for i in os.listdir(weight_path)]
        weights_available.sort()
        epoch = weights_available[-1]
    return epoch


if __name__ == '__main__':
    # Synthetic Shape Generation
    netG = init_netG()
    gen_weight_epoch = find_weight_epoch(gen_weight_path, gen_weight_epoch)
    netG_filename = f'{gen_weight_path}/netG_r{dim}_{gen_weight_epoch}.pth'
    print('Geneator weights to load:', netG_filename)
    netG.load_state_dict(torch.load(netG_filename))
    print('Geneator weights loaded')
    noises = torch.randn(n_samples, noise_dim)
    lst_samples = []
    for noise in torch.split(noises, gen_b_size):
        with torch.no_grad():
            samples = netG(noise.to(device))
            lst_samples.append(samples)
    output = torch.cat(lst_samples)
    del lst_samples
    print('Synthetic output shape:', output.shape)

    arr = output.cpu().numpy()
    for i in tqdm(range(arr.shape[0])):
        v = arr[i][0] > 0.3
        ax = plt.figure(figsize=(20, 20)).add_subplot(projection='3d')
        ax.view_init(elev, azim)
        ax.voxels(v)
        filename = f'model_{str(i).zfill(4)}_128'
        plt.savefig(os.path.join(save_path, filename), dpi=200)
        plt.close()

    print(f'{input_dim} plot done!')
