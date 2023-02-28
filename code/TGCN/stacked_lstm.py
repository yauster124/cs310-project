import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, p_dropout):
        super(StackedLSTM, self).__init__()
        self.weight1 = Parameter(torch.FloatTensor(input_size, hidden_size))
        self.bias1 = Parameter(torch.FloatTensor(hidden_size))
        self.lstm = nn.LSTM(input_size=hidden_size,
                          hidden_size=hidden_size,
                          bias=True,
                          batch_first=True,
                          dropout=0.2,
                          num_layers=2)
        self.relu = nn.ReLU()
        self.do = nn.Dropout(p_dropout)
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
        y = x @ self.weight1
        y = y + self.bias1
        y = self.relu(y)

        _, (y, _) = self.lstm(y)
        y = y[-1,:,:]
        y = self.do(y.squeeze())
        y = y @ self.weight2
        y = y + self.bias2

        return y