import matplotlib.pyplot as plt
from src.simple_upscaler import Upscaler
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
parser.add_argument('-e', '--num_epochs', type=int, required=False)
parser.add_argument('-n', '--num_models', type=int, required=False)
parser.add_argument('-b', '--batch_size', type=int, required=False)
parser.add_argument('-mb', '--mini_batch_size', type=int, required=False)
parser.add_argument('-id', '--input_dim', type=int, required=False)
parser.add_argument('-od', '--output_dim', type=int, required=False)
parser.add_argument('-dp', '--data_path', type=str, required=True)
parser.add_argument('-wp', '--weight_path', type=str, required=True)
parser.add_argument('--test', type=bool, required=False)
parser.add_argument('-b1', '--beta1', type=float, required=False)
parser.add_argument('-b2', '--beta2', type=float, required=False)
parser.add_argument('-lr', '--learning_rate', type=float, required=False)
#parser.add_argument('--test', type=bool, required=False)
args = parser.parse_args()

# argparse
input_dim = args.input_dim if args.input_dim else 64
output_dim = args.output_dim if args.output_dim else 256
num_models = args.num_models if args.num_models else 200
num_epochs = args.num_epochs if args.num_epochs else 100
batch_size = args.batch_size if args.batch_size else 4
mini_batch_size = args.mini_batch_size if args.mini_batch_size else 2
data_path = args.data_path
weights_path = args.weight_path
test = args.test if args.test else False

### HYPERPARAMETERS ###
# OPTIMIZER
lr = args.learning_rate if args.learning_rate else 1e-4
beta1 = args.beta1 if args.beta1 else 0.9
beta2 = args.beta2 if args.beta2 else 0.999

workers = 0
run_parallel = False
print('batch size:', batch_size)

# Set random seed for reproducibility
manualSeed = 42
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def import_data(data_path, num_models, input_dim, output_dim):
    # input models of resolution 64
    input_data_filename = f'{input_dim}'
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

    target_data_filename = f'{output_dim}'
    lst_binvox_target = []
    for file in tqdm(binvox_files):
        with open(os.path.join(data_path, target_data_filename, file), 'rb') as f:
            m = binvox_rw.read_as_3d_array(f).data
            lst_binvox_target.append(m)

    arr_input = np.array(lst_binvox_input)
    arr_target = np.array(lst_binvox_target)
    arr_input = arr_input.reshape(-1, 1, input_dim, input_dim, input_dim)
    arr_target = arr_target.reshape(-1, 1, output_dim, output_dim, output_dim)
    input_tensors = torch.from_numpy(arr_input).to(torch.float)
    target_tensors = torch.from_numpy(arr_target).to(torch.float)
    del arr_input, arr_target  # to empty memory

    return input_tensors, target_tensors


def get_dataloader(num_models, input_tensors, target_tensors):
    num_train = int(num_models * 0.8)
    train_dataset = torch.utils.data.TensorDataset(
        input_tensors[:num_train], target_tensors[:num_train])
    val_dataset = torch.utils.data.TensorDataset(
        input_tensors[num_train:], target_tensors[num_train:])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        # shuffle=True,
        shuffle=False,
        num_workers=workers,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=int(batch_size/2),
        # shuffle=True,
        shuffle=False,
        num_workers=workers,
    )

    return train_dataloader, val_dataloader


def init_upscaler(input_dim, output_dim):
    net = Upscaler(input_dim=input_dim, output_dim=output_dim)
    opt = optim.Adam(net.parameters(), lr=lr, betas=(beta1, beta2))
    criterion = torch.nn.BCELoss()
    # criterion = torch.nn.L1Loss()
    # criterion = torch.nn.MSELoss()
    return net, opt, criterion


def run(net, num_epochs, train_dataloader, val_dataloader, opt, criterion, input_dim, output_dim, device, start_epoch=0):
    os.makedirs(weights_path, exist_ok=True)
    # Training Loop
    print("Starting Training Loop...")
    lst_loss = []
    lst_val_loss = []
    # For each epoch
    for epoch in tqdm(range(start_epoch, start_epoch+num_epochs)):
        # For each batch in the dataloader
        for i, (input_data, target_data) in enumerate(train_dataloader):  # batch
            opt.zero_grad()  # make sure no grad recorded on opt before the start of epoch
            input_data_split = torch.split(input_data, mini_batch_size)
            target_data_split = torch.split(target_data, mini_batch_size)
            for j in range(len(input_data_split)):
                input_data_batch = input_data_split[j].to(device)
                target_data_batch = target_data_split[j].to(device)
                output = net(input_data_batch)
                err = criterion(output, target_data_batch)
                lst_loss.append(err.item())
                err.backward()  # err grad to opt
            opt.step()
            # opt.zero_grad()

            # validation
            val_input, val_target = next(iter(val_dataloader))
            val_input = val_input.to(device)
            with torch.no_grad():
                val_output = net(val_input)
                val_err = criterion(val_output.cpu(), val_target)
                lst_val_loss.append(val_err.item())

            # Output training stats at the end of epoch
            if i % 20 == 0:
                print(
                    f'[{epoch}/{num_epochs}] [{i}/{len(train_dataloader)}]\tLoss: {round(err.item(), 4)}\tVal loss: {round(val_err.item(), 4)}')

        if epoch % 5 == 0 and epoch != 0:
            # plot_convergence(G_losses, D_real_losses, D_fake_losses, real_accuracies, fake_accuracies)
            # save network weights
            net_filename = os.path.join(
                weights_path, f'net_r{input_dim}_r{output_dim}_{epoch}.pth')
            torch.save(net.state_dict(), net_filename)
            print('saved network weights', net_filename)


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print('device:', device)
# input_tensors, target_tensors = import_data(
#     data_path, num_models, input_dim, output_dim)
# train_dataloader, val_dataloader = get_dataloader(
#     num_models, input_tensors, target_tensors)
# net, opt, criterion = init_upscaler(
#     input_dim=input_dim, output_dim=output_dim)
# # summary(net, (batch_size, 1, 64, 64, 64))
# net = net.to(device)
# # file_net = '/gdrive/MyDrive/diss/weights/simple_upscaler/net_r64_r256_e5_weights.pth'
# # net.load_state_dict(torch.load(file_net))
# run(net, num_epochs, train_dataloader, val_dataloader,
#     opt, criterion, input_dim, output_dim, device)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)
    net, opt, criterion = init_upscaler(
        input_dim=input_dim, output_dim=output_dim)
    net = net.to(device)
    # summary(net, (batch_size, 1, 64, 64, 64))
    if test:
        weights_available = [i.strip('.pth').split('_')[-1]
                             for i in os.listdir(weights_path)]
        weights_available.sort()
        last_weights = weights_available[-1]
        net_filename = f'net_r{input_dim}_r{output_dim}_{last_weights}.pth'
        print('weights to load:', net_filename)
        net.load_state_dict(torch.load(
            os.path.join(weights_path, net_filename)))
        print('loaded weights on net with', net_filename)
        input_tensors, target_tensors = import_data(
            data_path, 5, input_dim, output_dim)
        output = net(input_tensors.to(device))
        output = output.numpy()
        with open('test.npy', 'wb') as f:
            np.save(f, output)
    else:
        input_tensors, target_tensors = import_data(
            data_path, num_models, input_dim, output_dim)
        train_dataloader, val_dataloader = get_dataloader(
            num_models, input_tensors, target_tensors)
        run(net, num_epochs, train_dataloader, val_dataloader,
            opt, criterion, input_dim, output_dim, device)
