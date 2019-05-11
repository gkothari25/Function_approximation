#Author:- Gaurav Kothyari

import torch
import torch.nn.functional as F
from collections import OrderedDict
from torch import optim
import numpy as np


class Orfunction(torch.nn.Module):

    def __init__(self, D_in, H, D_out):
        super().__init__()
        self.D_in = D_in
        self.H = H
        self.D_out = D_out
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = self.linear1(x)
        x = F.tanh(x)
        x = self.linear2(x)
        x = F.tanh(x)
        x = torch.sigmoid(x)
        return x


D_in = 2
H = 2
D_out = 1

function_network =  Orfunction(D_in,H,D_out)


loss_function = torch.nn.L1Loss()

variables = function_network.parameters()

optimizer = optim.SGD(variables, lr=0.0001)

x = torch.tensor([[0,0],[0,1],[1,0]],dtype=torch.float)
y = torch.tensor([[0], [1], [1]], dtype=torch.float)

epochs = 1000000

for i in range(epochs):
    print('iteration', i)
    optimizer.zero_grad()
    output = function_network(x)
    loss = loss_function(output,y)
    loss.backward()
    print("loss",loss.item())
    optimizer.step()

test_t =torch.tensor([[1,1],[0,0],[1,1]],dtype=torch.float)


result = function_network.forward(test_t)

print('probability',result)
output = np.zeros(result.shape)

for i in range(result.shape[0]):
    if result[i] >= 0.5:
        output[i] = 1
    else:
        output[i] = 0
print("output is ", output)


