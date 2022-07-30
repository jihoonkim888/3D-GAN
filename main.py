import os
import numpy as np
import torch
from src.GAN import Generator
import argparse
from src.upscaler import Upscaler
import argparse
from src import binvox_rw
from tqdm.auto import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--n_samples', type=int, required=True)
parser.add_argument('-gwp', '--gen_weight_path', type=str, required=True)
parser.add_argument('-gwe', '--gen_weight_epoch', type=int, required=False)
parser.add_argument('-uwp', '--upscaler_weight_path', type=str, required=True)
parser.add_argument('-uwe', '--upscaler_weight_epoch',
                    type=int, required=False)
parser.add_argument('-sp', '--save_path', type=str, required=True)
parser.add_argument('-gb', '--gen_batch_size', type=int, required=False)
parser.add_argument('-ub', '--upscaler_batch_size', type=int, required=False)
parser.add_argument('-v', '--visdom', type=bool, required=False)
parser.add_argument('-p', '--port', type=int, required=False)
args = parser.parse_args()

# argparse
gen_weight_path = args.gen_weight_path
gen_weight_epoch = args.gen_weight_epoch if args.gen_weight_epoch else None
upscaler_weight_path = args.upscaler_weight_path
upscaler_weight_epoch = args.upscaler_weight_epoch if args.upscaler_weight_epoch else None
save_path = args.save_path
n_samples = args.n_samples
visdom = args.visdom if args.visdom else False
port = args.port if args.port else None

# parameters
dim = 64
input_dim = 64
output_dim = 128
noise_dim = 200
gen_b_size = args.gen_batch_size if args.gen_batch_size else 50
upscaler_b_size = args.upscaler_batch_size if args.upscaler_batch_size else 2
conv_channels = 256

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)


def init_netG():
    netG = Generator(in_channels=conv_channels, out_dim=dim,
                     out_channels=1, noise_dim=noise_dim)
    netG = netG.to(device)
    return netG


def init_upscaler(input_dim, output_dim):
    net = Upscaler(input_dim=input_dim, output_dim=output_dim)
    return net


def find_weight_epoch(weight_path, weight_epoch):
    if weight_epoch is not None:
        epoch = weight_epoch
    else:
        weights_available = [i.strip('.pth').split('_')[-1]
                             for i in os.listdir(weight_path)]
        weights_available.sort()
        epoch = weights_available[-1]
    return epoch


def getVFByMarchingCubes(voxels, threshold=0.5):
    v, f = sk.marching_cubes(voxels, level=threshold, method='_lorensen')
    return v, f


def plotVoxelVisdom(voxels, visdom, title):
    v, f = getVFByMarchingCubes(voxels)
    visdom.mesh(X=v, Y=f, opts=dict(opacity=0.5, title=title))


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
    print('Synthetic output shape:', output.shape)

    torch.cuda.empty_cache()

    # Upscaling
    net = init_upscaler(
        input_dim=input_dim, output_dim=output_dim)
    net = net.to(device)
    upscaler_weight_epoch = find_weight_epoch(
        upscaler_weight_path, upscaler_weight_epoch)
    net_filename = f'net_r{input_dim}_r{output_dim}_{upscaler_weight_epoch}.pth'
    print('Upscaler weights to load:', net_filename)
    net.load_state_dict(torch.load(
        os.path.join(upscaler_weight_path, net_filename)))
    print('Upscaler weights loaded')
    lst_samples = []
    for t in torch.split(output, upscaler_b_size):
        with torch.no_grad():
            samples = net(t.to(device))
            lst_samples.append(samples)
    output = torch.cat(lst_samples)
    del lst_samples
    output = output.cpu().numpy()
    print('Output shape:', output.shape)
    with open(save_path, 'wb') as f:
        np.save(f, output)
    print('Output saved at', save_path)

    if visdom:
        import skimage.measure as sk
        import visdom
        vis = visdom.Visdom(port=port)

        for i in tqdm(range(output.shape[0])):
            a = output[i][0]
            filename = 'Model '+str(i)
            plotVoxelVisdom(a, vis, filename)
