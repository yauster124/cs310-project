import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.jit as jit
from collections import namedtuple
from typing import List, Tuple
from torch import Tensor


class CustomGRUCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(CustomGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w_r = Parameter(torch.FloatTensor(input_size, hidden_size))
        self.w_z = Parameter(torch.FloatTensor(input_size, hidden_size))
        self.w_h = Parameter(torch.FloatTensor(input_size, hidden_size))
        self.u_r = Parameter(torch.FloatTensor(hidden_size, hidden_size))
        self.u_z = Parameter(torch.FloatTensor(hidden_size, hidden_size))
        self.u_h = Parameter(torch.FloatTensor(hidden_size, hidden_size))

        self.b_r = Parameter(torch.FloatTensor(hidden_size))
        self.b_z = Parameter(torch.FloatTensor(hidden_size))
        self.b_h = Parameter(torch.FloatTensor(hidden_size))

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    @jit.script_method
    def forward(self, input, hx):
        r_t = self.sigmoid((self.w_r @ input) + (self.u_r @ hx) + self.b_r)
        z_t = self.sigmoid()
        n_t = 

        return hy, (hy, cy)