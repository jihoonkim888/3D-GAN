import os
import numpy as np
from tqdm.auto import tqdm
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from src import binvox_rw
from src.GAN import Discriminator, Generator, weights_init
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--num_epochs', type=int, required=False)
parser.add_argument('-n', '--num_models', type=int, required=False)
parser.add_argument('-b', '--batch_size', type=int, required=False)
parser.add_argument('-mb', '--mini_batch_size', type=int, required=False)
parser.add_argument('-d', '--dim', type=int, required=False)
parser.add_argument('-nd', '--noise_dim', type=int, required=False)
parser.add_argument('-ch', '--conv_channels', type=int, required=False)
parser.add_argument('-dp', '--data_path', type=str, required=True)
parser.add_argument('-wp', '--weight_path', type=str, required=True)
parser.add_argument('-b1', '--beta1', type=float, required=False)
parser.add_argument('-b2', '--beta2', type=float, required=False)
parser.add_argument('-lrg', '--learning_rate_G', type=float, required=False)
parser.add_argument('-lrd', '--learning_rate_D', type=float, required=False)
parser.add_argument('--run_parallel', type=bool, required=False)
# parser.add_argument('-a', '--alpha', type=float,
#                     required=False, help='alpha for weighted BCE loss')
args = parser.parse_args()

# argparse
dim = args.dim if args.dim else 64
num_models = args.num_models if args.num_models else 2000
num_epochs = args.num_epochs if args.num_epochs else 100
batch_size = args.batch_size if args.batch_size else 100
mini_batch_size = args.mini_batch_size if args.mini_batch_size else 50
data_path = args.data_path
weights_path = args.weight_path
lr_G = args.learning_rate_G if args.learning_rate_G else 0.0025
lr_D = args.learning_rate_D if args.learning_rate_D else 1e-5
beta1 = args.beta1 if args.beta1 else 0.5
noise_dim = args.noise_dim if args.noise_dim else 200  # latent space vector dim
conv_channels = args.conv_channels if args.conv_channels else 256
run_parallel = args.run_parallel if args.run_parallel else False
num_split = int(batch_size / mini_batch_size)
print('batch size:', batch_size, 'mini batch:',
      mini_batch_size, 'num_split:', num_split)

workers = 0
# Set random seed for reproducibility
manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)
os.makedirs(weights_path, exist_ok=True)


def import_data(data_path, num_models, dim):
    # input models of resolution 64
    input_data_filename = f'{dim}'
    binvox_files = os.listdir(os.path.join(data_path, input_data_filename))
    binvox_files.sort()
    binvox_files = binvox_files[:num_models]
    print('number of binvox files:', len(binvox_files))
    print(binvox_files[0], binvox_files[-1])

    lst_binvox_input = []
    for file in tqdm(binvox_files):
        with open(os.path.join(data_path, input_data_filename, file), 'rb') as f:
            m = binvox_rw.read_as_3d_array(f).data
            lst_binvox_input.append(m)

    arr_input = np.array(lst_binvox_input).reshape(-1, 1, dim, dim, dim)
    arr_input = torch.from_numpy(arr_input).to(torch.float)
    return arr_input


def get_dataloader(input_tensors):
    dataloader = DataLoader(
        input_tensors,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
    )
    return dataloader


def init_GAN():
    netG = Generator(in_channels=conv_channels, out_dim=dim,
                     out_channels=1, noise_dim=noise_dim)
    if run_parallel:
        netG = torch.nn.DataParallel(netG)
    netG = netG.to(device)
    netG.apply(weights_init)
    netD = Discriminator(
        in_channels=1, out_conv_channels=conv_channels, dim=dim)
    if run_parallel:
        netD = torch.nn.DataParallel(netD)
    netD = netD.to(device)
    netD.apply(weights_init)
    # Setup Adam optimizers for both G and D
    optG = optim.Adam(netG.parameters(), lr=lr_G, betas=(beta1, 0.999))
    optD = optim.Adam(netD.parameters(), lr=lr_D, betas=(beta1, 0.999))
    criterion = torch.nn.BCELoss()
    return netG, netD, optG, optD, criterion


def plot_convergence(G_losses, D_real_losses, D_fake_losses, real_accuracies, fake_accuracies):
    lst_epoch = np.array(range(len(G_losses)))
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(lst_epoch, G_losses, label="G")
    plt.plot(lst_epoch, D_real_losses, label="D_real")
    plt.plot(lst_epoch, D_fake_losses, label="D_fake")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    filename = '3D_GAN_loss_plot.png'
    plt.savefig(os.path.join(weights_path, filename), dpi=200)
    print('Loss plot saved to', filename)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.title("Discriminator Accuracies During Training")
    plt.plot(lst_epoch, real_accuracies, label="acc_real")
    plt.plot(lst_epoch, fake_accuracies, label="acc_fake")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    filename = '3D_GAN_acc_plot.png'
    plt.savefig(os.path.join(weights_path, filename), dpi=200)
    print('Accuracy plot saved to', filename)
    plt.close()

    return


def run(dataloader, netG, netD, optG, optD, criterion):
    # Lists to keep track of progress
    G_losses = []
    D_real_losses = []
    D_fake_losses = []
    real_accuracies = []
    fake_accuracies = []

    ##### START OF EPOCH #####
    for epoch in tqdm(range(num_epochs)):
        # append average of errors and accuracies after every epoch
        lst_errD_real_batch = []
        lst_errD_fake_batch = []
        lst_errG_batch = []
        lst_train_acc_real_batch = []
        lst_train_acc_fake_batch = []
        lst_update = []

        ### START OF BATCH ###
        for data_all in dataloader:
            ### START OF DISCRIMINATOR UPDATE ###
            data_split = torch.split(data_all, mini_batch_size)

            lst_errD_real_mini = []
            lst_errD_fake_mini = []
            lst_train_acc_real_mini = []
            lst_train_acc_fake_mini = []
            lst_errG_mini = []

            optD.zero_grad()
            ## START OF MINI ##
            for real_data in data_split:
                # Discriminator on real data #
                real_data = real_data.to(device)
                label_real = torch.full(
                    (mini_batch_size,), 1.0, dtype=torch.float, device=device)

                outD_real = netD(real_data).view(-1)
                errD_real = criterion(outD_real, label_real) / num_split
                errD_real.backward()
                lst_errD_real_mini.append(errD_real.item())

                # D acc for real samples
                train_acc_real = (torch.sum((outD_real > 0.5).to(
                    int) == label_real) / mini_batch_size).item()
                lst_train_acc_real_mini.append(train_acc_real)

                # Update Discriminator with fake data generated from noise #
                label_fake = torch.full(
                    (mini_batch_size,), 0.0, dtype=torch.float, device=device)
                noise = torch.randn(mini_batch_size, noise_dim, device=device)
                fake = netG(noise).detach()
                outD_fake = netD(fake).view(-1)
                errD_fake = criterion(outD_fake, label_fake) / num_split
                errD_fake.backward()
                lst_errD_fake_mini.append(errD_fake.item())

                # D acc for samples from G
                train_acc_fake = (torch.sum((outD_fake > 0.5).to(
                    int) == label_fake) / mini_batch_size).item()
                lst_train_acc_fake_mini.append(train_acc_fake)
            ## END OF MINI ##

            ### START OF BATCH ###
            # update D only if classification acc is less than 80% for stability
            lst_errD_fake_batch.append(np.sum(lst_errD_fake_mini))
            lst_errD_real_batch.append(np.sum(lst_errD_real_mini))
            lst_train_acc_real_batch.append(np.mean(lst_train_acc_real_mini))
            lst_train_acc_fake_batch.append(np.mean(lst_train_acc_fake_mini))

            acc_real_mean = np.mean(lst_train_acc_real_mini)
            acc_fake_mean = np.mean(lst_train_acc_fake_mini)
            update = ((acc_real_mean + acc_fake_mean) / 2) < 0.8
            lst_update.append(update)
            if update:
                optD.step()
            optD.zero_grad()
            ### END OF DISCRIMINATOR UPDATE ###

            ### START OF GENERATOR UPDATE ###
            optG.zero_grad()

            ## START OF MINI ##
            for _ in range(num_split):
                label = torch.full((mini_batch_size,), 1.0,
                                   dtype=torch.float, device=device)
                noise = torch.randn(mini_batch_size, noise_dim, device=device)
                fake = netG(noise)
                output = netD(fake).view(-1)
                errG = criterion(output, label) / num_split
                errG.backward()
                lst_errG_mini.append(errG.item())
            ## END OF MINI ##

            ### START OF BATCH ###
            lst_errG_batch.append(np.mean(lst_errG_mini))
            optG.step()
            optG.zero_grad()
            ### END OF BATCH ###
            ### END OF GENERATOR UPDATE ###

        ##### START OF EPOCH #####
        G_losses.append(np.sum(lst_errG_batch))
        D_real_losses.append(np.sum(lst_errD_real_batch))
        D_fake_losses.append(np.sum(lst_errD_fake_batch))
        real_accuracies.append(np.mean(lst_train_acc_real_batch))
        fake_accuracies.append(np.mean(lst_train_acc_fake_batch))
        # print('lst_train_acc_real_batch:', lst_train_acc_real_batch,
        #       'lst_train_acc_fake_batch:', lst_train_acc_fake_batch)

        print(f'[{epoch}/{num_epochs}]\tLoss_D_real: {round(D_real_losses[epoch], 4)}\tLoss_D_fake: {round(D_fake_losses[epoch], 4)}\tLoss_G: {round(G_losses[epoch], 4)}\tacc_D(x): {round(real_accuracies[epoch], 4)}\tacc_D(G(z)): {round(fake_accuracies[epoch], 4)}\tupdate: {np.sum(lst_update)}/{len(lst_update)}')

        # save net weights every 5 epochs
        if epoch % 5 == 0 and epoch != 0:
            # save network weights
            netG_filename = f'{weights_path}/netG_r{dim}_{epoch}.pth'
            netD_filename = f'{weights_path}/netD_r{dim}_{epoch}.pth'
            torch.save(netG.state_dict(), netG_filename)
            torch.save(netD.state_dict(), netD_filename)
            print('saved network weights', netG_filename)

            plot_convergence(G_losses, D_real_losses,
                             D_fake_losses, real_accuracies, fake_accuracies)

        #### END OF EPOCH #####

    return


if __name__ == '__main__':
    netG, netD, optG, optD, criterion = init_GAN()
    # print("\n\nGenerator summary\n\n")
    # summary(netG, (1, noise_dim))
    # print("\n\nDiscriminator summary\n\n")
    # summary(netD, (mini_batch_size, 1, dim, dim, dim))

    print("Importing real data...")
    input_tensors = import_data(data_path, num_models, dim)
    print("Done!")
    print("Initialising dataloader...")
    dataloader = get_dataloader(input_tensors)
    print("Done!")
    print("Starting Training Loop...")
    run(dataloader, netG, netD, optG, optD, criterion)
