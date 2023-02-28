import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class SingleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, p_dropout):
        super(SingleGRU, self).__init__()
        self.weight1 = Parameter(torch.FloatTensor(input_size, hidden_size))
        self.bias1 = Parameter(torch.FloatTensor(hidden_size))
        # FOR gru with initial linear layer
        # self.gru1 = nn.GRU(input_size=hidden_size,
        #                   hidden_size=hidden_size,
        #                   bias=True,
        #                   batch_first=True)
        # FOR gru without initial linear layer
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          bias=True,
                          batch_first=True)
        self.relu = nn.ReLU()
        self.do = nn.Dropout(0.3)
        self.weight2 = Parameter(torch.FloatTensor(hidden_size, num_classes))
        self.bias2 = Parameter(torch.FloatTensor(num_classes))
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.weight1.size(1))
        self.weight1.data.uniform_(-stdv1, stdv1)
        self.bias1.data.uniform_(-stdv1, stdv1)

        stdv2 = 1. / math.sqrt(self.weight2.size(1))
        self.weight2.data.uniform_(-stdv2, stdv2)
        self.bias2.data.uniform_(-stdv2, stdv2)
    
    def forward(self, x):
        # WITH INITIAL LINEAR LAYER
        # y = x @ self.weight1
        # y = y + self.bias1
        # y = self.relu(y)

        # _, y = self.gru1(y)
        # y = self.do(y.squeeze())
        # y = y @ self.weight2
        # y = y + self.bias2

        # WITHOUT INITIAL LINEAR LAYER
        _, y = self.gru(x)
        y = self.do(y.squeeze())
        y = y @ self.weight2
        y = y + self.bias2

        return y