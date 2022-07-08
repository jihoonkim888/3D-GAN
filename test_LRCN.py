from src.LRCN import LRCN
import torch
from torchsummary import summary

dim = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)

net = LRCN()
print("\n\nGenerator summary\n\n")
summary(net, (1, dim, dim, dim))
