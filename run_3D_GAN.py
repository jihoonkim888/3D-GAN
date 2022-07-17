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
parser.add_argument('--synthesise', type=bool, required=False)
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
synthesise = args.synthesise if args.synthesise else False
lr_G = args.learning_rate_G if args.learning_rate_G else 0.0025
lr_D = args.learning_rate_D if args.learning_rate_D else 1e-5
beta1 = args.beta1 if args.beta1 else 0.5
workers = 0

noise_dim = args.noise_dim if args.noise_dim else 200  # latent space vector dim
conv_channels = args.conv_channels if args.conv_channels else 256
run_parallel = args.run_parallel if args.run_parallel else False
k = int(batch_size / mini_batch_size)
print('batch size:', batch_size, 'mini batch:', mini_batch_size, 'k:', k)

# Set random seed for reproducibility
manualSeed = 42
# manualSeed = random.randint(1, 10000) # use if you want new results
# print("Random Seed: ", manualSeed)
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
    #    train_dataset = torch.utils.data.TensorDataset(
    #        input_tensors)

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

    criterion = torch.nn.BCELoss()
    # Setup Adam optimizers for both G and D
    optG = optim.Adam(netG.parameters(), lr=lr_G, betas=(beta1, 0.999))
    optD = optim.Adam(netD.parameters(), lr=lr_D, betas=(beta1, 0.999))

    return netG, netD, optG, optD, criterion


def plot_convergence(G_losses, D_real_losses, D_fake_losses, real_accuracies, fake_accuracies):
    lst_epoch = np.array(range(int(num_epochs * num_models /
                         mini_batch_size))) / len(dataloader)
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(lst_epoch, G_losses, label="G")
    plt.plot(lst_epoch, D_real_losses, label="D_real")
    plt.plot(lst_epoch, D_fake_losses, label="D_fake")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    filename = os.path.join(data_path, '3D_GAN_loss_plot.png')
    plt.savefig(filename, dpi=200)
    print('Loss plot saved to', filename)

    plt.figure(figsize=(10, 5))
    plt.title("Discriminator Accuracies During Training")
    plt.plot(lst_epoch, real_accuracies, label="acc_real")
    plt.plot(lst_epoch, fake_accuracies, label="acc_fake")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    filename = os.path.join(data_path, '3D_GAN_syn_acc_plot.png')
    plt.savefig(filename, dpi=200)
    print('Accuracy plot saved to', filename)
    return


def run(dataloader, netG, netD, optG, optD, criterion):

    # Lists to keep track of progress
    G_losses = []
    D_real_losses = []
    D_fake_losses = []
    real_accuracies = []
    fake_accuracies = []
    start_epoch = 0
    iters = 0

    real_label = 1.
    fake_label = 0.

    # Training Loop
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        lst_train_acc_real = []
        lst_train_acc_fake = []
        for i, data_all in enumerate(dataloader, 0):
            data_split = torch.split(data_all, mini_batch_size)
            optD.zero_grad()
    #         print('reset netD grads')
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with all-real batch
            # Format batch
            for j in range(len(data_split)):
                real_data = data_split[j]
                real_data = real_data.to(device)
                b_size = real_data.size(0)
                label_real = torch.full(
                    (b_size,), real_label, dtype=torch.float, device=device)
                label_fake = torch.full(
                    (b_size,), fake_label, dtype=torch.float, device=device)
                # Forward pass real batch through D
                outD_real = netD(real_data).view(-1)
                # Calculate loss on all-real batch
                D_x = outD_real.mean().item()
                train_acc_real = torch.sum(
                    (outD_real > 0.5).to(int) == label_real) / b_size
                lst_train_acc_real.append(train_acc_real.item())
                errD_real = criterion(outD_real, label_real) / len(data_split)
                errD_real.backward()

                # Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, noise_dim, device=device)
                # Generate fake image batch with G
                fake = netG(noise).detach()
                outD_fake = netD(fake).view(-1)
                D_G_z1 = outD_fake.mean().item()
                train_acc_fake = torch.sum(
                    (outD_fake > 0.5).to(int) == label_fake) / b_size
                lst_train_acc_fake.append(train_acc_fake.item())
                errD_fake = criterion(outD_fake, label_fake) / len(data_split)
                errD_fake.backward()

                errD = errD_real + errD_fake

            # update D only if classification acc is less than 80% for stability
    #         if (i+1) % k == 0 or (i+1) == len(dataloader):
                if j == len(data_split)-1:
                    acc_real_mean = np.mean(lst_train_acc_real)
                    acc_fake_mean = np.mean(lst_train_acc_fake)
                    update = ((acc_real_mean + acc_fake_mean) / 2) < 0.8
                    if update:
                        optD.step()  # update the weights only after accumulating k small batches
    #                     print('updated optD')

                    optD.zero_grad()  # reset gradients for accumulation for the next large batch
                    lst_train_acc_real = []
                    lst_train_acc_fake = []

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            # fake labels are real for generator cost
            optG.zero_grad()
    #         print('reset netG grads')
            for j in range(len(data_split)):
                label = torch.full((b_size,), real_label,
                                   dtype=torch.float, device=device)
                # Since we just updated D, perform another forward pass of all-fake batch through D
                fake = netG(noise)
                output = netD(fake).view(-1)
                errG = criterion(output, label) / len(data_split)
                errG.backward()

                D_G_z2 = output.mean().item()
    #             if (i+1) % k == 0 or (i+1) == len(dataloader):
                if j == len(data_split)-1:
                    optG.step()  # update the weights only after accumulating k small batches
                    optG.zero_grad()  # reset gradients for accumulation for the next large_batch
    #                 print('updated optG')

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_fake_losses.append(errD_fake.item())
                D_real_losses.append(errD_real.item())
                fake_accuracies.append(train_acc_fake.item())
                real_accuracies.append(train_acc_real.item())

            # # Output training stats
            # if i % 10 == 0:  # print progress every epoch
            #     print(f'[{epoch}/{start_epoch+num_epochs}][{i}/{len(dataloader)}]\tLoss_D: {round(errD.item(), 4)}\tLoss_G: {round(errG.item(), 4)}\tD(x): {round(D_x, 4)}\tD(G(z)): {round(D_G_z1, 4)} / {round(D_G_z2, 4)}')

            iters += 1

        print(f'[{epoch}/{start_epoch+num_epochs}][{i}/{len(dataloader)}]\tLoss_D: {round(errD.item(), 4)}\tLoss_G: {round(errG.item(), 4)}\tD(x): {round(D_x, 4)}\tD(G(z)): {round(D_G_z1, 4)} / {round(D_G_z2, 4)}\tD(x) acc: {round(acc_real_mean, 4)}\tD(G(z)) acc: {round(acc_fake_mean, 4)}')

        # save net weights every 10 epochs
        if epoch % 10 == 0 and epoch != 0:
            # save network weights
            netG_filename = f'{weights_path}/netG_r{dim}_{epoch}.pth'
            netD_filename = f'{weights_path}/netD_r{dim}_{epoch}.pth'
            torch.save(netG.state_dict(), netG_filename)
            torch.save(netD.state_dict(), netD_filename)
            print('saved network weights', netG_filename)

    return G_losses, D_real_losses, D_fake_losses, real_accuracies, fake_accuracies


if __name__ == '__main__':
    netG, netD, optG, optD, criterion = init_GAN()
    # print("\n\nGenerator summary\n\n")
    # summary(netG, (1, noise_dim))
    # print("\n\nDiscriminator summary\n\n")
    # summary(netD, (mini_batch_size, 1, dim, dim, dim))

    if synthesise:
        weights_available = [i.strip('.pth').split('_')[-1]
                             for i in os.listdir(weights_path)]
        weights_available.sort()
        last_weights = weights_available[-1]
        netG_filename = f'{weights_path}/netG_r{dim}_{last_weights}.pth'
        netD_filename = f'{weights_path}/netD_r{dim}_{last_weights}.pth'
        print('weights to load:', netG_filename, netD_filename)
        netG.load_state_dict(torch.load(netG_filename))
        netD.load_state_dict(torch.load(netD_filename))
        print('weights loaded')
        b_size = 20
        noise = torch.randn(b_size, noise_dim, device=device)
        with torch.no_grad():
            output = netG(noise)
        output = output.cpu().numpy()
        output_filename = os.path.join(data_path, 'test.npy')
        print('synthetic output:', output_filename)
        with open(output_filename, 'wb') as f:
            np.save(f, output)
        print('Done!')

    else:
        input_tensors = import_data(data_path, num_models, dim)
        dataloader = get_dataloader(input_tensors)
        G_losses, D_real_losses, D_fake_losses, real_accuracies, fake_accuracies = run(
            dataloader, netG, netD, optG, optD, criterion)
        plot_convergence(G_losses, D_real_losses, D_fake_losses,
                         real_accuracies, fake_accuracies)
