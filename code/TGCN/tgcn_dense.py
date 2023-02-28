import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from tgcn_model import GraphConvolution_att


class GCN_2Layer(nn.Module):
    """2-Layer GCN model specified in https://arxiv.org/abs/1811.05320"""

    def __init__(self, in_features, p_dropout):
        super(GCN_2Layer, self).__init__()
        self.gc1 = GraphConvolution_att(in_features, in_features)
        self.bn1 = nn.BatchNorm1d(55 * in_features)

        self.gc2 = GraphConvolution_att(in_features, in_features)
        self.bn2 = nn.BatchNorm1d(55 * in_features)

        self.do = nn.Dropout(p_dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.relu(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.sigmoid(y)
        y = self.do(y)

        return y + x

class TGCN_GRU(nn.Module):
    def __init__(self, in_features, p_dropout):
        super(TGCN_GRU, self).__init__()
        self.gcn = GCN_2Layer(in_features, p_dropout)
        self.gru = nn.GRU(input_size=110,
                          hidden_size=110,
                          num_layers=1,
                          bias=True)
        self.do = nn.Dropout1d(0.3)
        self.dense = nn.Linear(110, 100)    
    
    def forward(self, x):
        features = self.gcn(x)
        features = torch.stack(torch.chunk(features, 50, 2))    # Separate into different tensors for each timestep.
        features = torch.flatten(features, start_dim=2)     # Flatten features to pass into GRU.
        print(features.shape)
        y = self.do(features)
        y = self.dense(y)
        y = torch.mean(y, dim=0)

        return y
