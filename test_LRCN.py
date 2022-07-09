from src.LRCN import LRCN, weights_init
import torch
from torchinfo import summary

dim = 64
c = 5
input_shape = (1, 1, c, dim, dim)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)
print('input shape:', input_shape)

net = LRCN(input_dim=dim, kernel_size=3, output_dim=128, hidden_size=100)
net.apply(weights_init)
print("\n\nGenerator summary\n\n")
summary(net, input_shape)
