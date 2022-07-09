from src.LRCN import LRCN, weights_init
import torch
from torchinfo import summary

input_dim = 64
output_dim = 128
c = 5
input_shape = (1, 1, input_dim, input_dim, input_dim)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)
print('input shape:', input_shape)

net = LRCN(input_dim=input_dim, kernel_size=3,
           output_dim=output_dim, hidden_size=1000)
net.apply(weights_init)
print("\n\nGenerator summary\n\n")
summary(net, input_shape)

input = torch.randn((5, 1, input_dim, input_dim, input_dim))
with torch.no_grad():
    output = net(input)
    print(output.shape)
    # print(output)
