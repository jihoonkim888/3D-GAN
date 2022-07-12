import argparse
from src import binvox_rw
from src.LRCN import LRCN, weights_init
from torch.utils.data import DataLoader
from torchinfo import summary
import torch.optim as optim
import torch
import random
from tqdm.auto import tqdm
import numpy as np
import os
# os.system('pip install torchinfo')

# import matplotlib.pyplot as plt
# import matplotlib.animation as animation


parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, required=False)
parser.add_argument('--num_models', type=int, required=False)
parser.add_argument('--batch_size', type=int, required=False)
parser.add_argument('--input_dim', type=int, required=False)
parser.add_argument('--output_dim', type=int, required=False)
parser.add_argument('--c', type=int, required=False)
parser.add_argument('--data_path', type=str, required=False)
args = parser.parse_args()


data_path = args.data_path if args.data_path else os.path.join(
    '..', 'data', 'shapenet-lamp')

### HYPERPARAMETERS ###
# OPTIMIZER
lr = 1e-4
beta1 = 0.5
beta2 = 0.999

# argparse
input_dim = args.input_dim if args.input_dim else 64
output_dim = args.output_dim if args.output_dim else 128
num_models = args.num_models if args.num_models else 500
c = args.c if args.c else 5
num_epochs = args.num_epochs if args.num_epochs else 100
batch_size = args.batch_size if args.batch_size else 4
workers = 0
run_parallel = False
print('batch size:', batch_size)

# Set random seed for reproducibility
manualSeed = 42
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def import_data(num_models, input_dim, output_dim):
    # input models of resolution 64
    input_data_filename = f'shapenet-lamp-binvox-{input_dim}'
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

    target_data_filename = f'shapenet-lamp-binvox-{output_dim}'
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
        batch_size=batch_size,
        # shuffle=True,
        shuffle=False,
        num_workers=workers,
    )

    return train_dataloader, val_dataloader


def init_LRCN(batch_size, input_dim, output_dim, c, device):

    net = LRCN(input_dim=input_dim, kernel_size=3, c=c,
               output_dim=output_dim, batch_size=batch_size)
    net = net.to(device)
    opt = optim.Adam(net.parameters(), lr=lr, betas=(beta1, beta2))
    # criterion = torch.nn.BCELoss()
    criterion = torch.nn.L1Loss()
    # criterion = torch.nn.MSELoss()
    net.apply(weights_init)

    # input_shape = (batch_size, 1, input_dim, input_dim, input_dim)
    # print('input shape:', input_shape)
    # print("\n\nNetwork summary\n\n")
    # summary(net, input_shape)

    return net, opt, criterion


def run(net, num_epochs, train_dataloader, val_dataloader, opt, criterion, input_dim, output_dim, device, start_epoch=0):
    weights_path = 'weights/LRCN'
    os.makedirs(weights_path, exist_ok=True)
    # Training Loop
    print("Starting Training Loop...")
    lst_loss = []
    lst_val_loss = []
    # For each epoch
    for epoch in tqdm(range(start_epoch, start_epoch+num_epochs)):
        # For each batch in the dataloader
        for i, (input_data, target_data) in enumerate(train_dataloader):  # batch
            input_data = input_data.to(device)
            opt.zero_grad()  # make sure no grad recorded on opt before the start of epoch
            # get inference
            for batch in range(input_data.size(0)):
                input_data_single = input_data[batch].reshape(
                    1, 1, input_dim, input_dim, input_dim)
                output = net(input_data_single)
                # Calculate loss
                err = criterion(output.cpu(), target_data)
                # lst_loss.append(err.item())
                err.backward()  # err grad to opt
            opt.step()
            opt.zero_grad()

            # validation
            val_input, val_target = next(iter(val_dataloader))
            val_input = val_input.to(device)
            with torch.no_grad():
                val_output = net(val_input)
                val_err = criterion(val_output.cpu(), val_target)
                lst_val_loss.append(val_err)

            # Output training stats at the end of epoch
            if i % 20 == 0:
                print(
                    f'[{epoch}/{num_epochs}] [{i}/{len(train_dataloader)}]\tLoss: {round(err.item(), 4)}\tVal loss: {round(val_err.item(), 4)}')

        if epoch % 5 == 0 and epoch != 0:
            # plot_convergence(G_losses, D_real_losses, D_fake_losses, real_accuracies, fake_accuracies)
            # save network weights
            net_filename = os.path.join(
                weights_path, f'net_r{input_dim}_r{output_dim}_e{epoch}_weights.pth')
            torch.save(net.state_dict(), net_filename)
            print('saved network weights', net_filename)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)
    # input_tensors, target_tensors = import_data(
    #     num_models, input_dim, output_dim)
    # train_dataloader, val_dataloader = get_dataloader(
    #     num_models, input_tensors, target_tensors)
    net, opt, criterion = init_LRCN(
        batch_size=1, input_dim=input_dim, output_dim=output_dim, c=c, device=device)
    input_sample = torch.randn(
        1, 1, input_dim, input_dim, input_dim).to(device)
    output_sample = net(input_sample)
    print('output size:', output_sample.size())
    # run(net, num_epochs, train_dataloader, val_dataloader,
    #     opt, criterion, input_dim, output_dim, device)
