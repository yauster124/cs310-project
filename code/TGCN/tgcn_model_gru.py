import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from tgcn_model import GC_Block

# class GraphConvolution(nn.Module):
#     """
#     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
#     """

#     def __init__(self, in_features, out_features):
#         """
#         in_features: Number of features from the previous layer
#             - Or if there is no previous layer, then it's the length of the flattened coordinates (usually 2*num_samples)
#         out_features: Number of features we want to output from HW
#         """
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.FloatTensor(50, in_features, out_features))
#         self.att = Parameter(torch.FloatTensor(50, 55, 55))
#         self.bias = Parameter(torch.FloatTensor(50, 1, out_features))
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         self.att.data.uniform_(-stdv, stdv)
#         self.bias.data.uniform_(-stdv, stdv)

#     def forward(self, input):
#         # H (input): (batch_size, num_keypoints, in_features)
#         #   - We store a set number of features (in_features) for each keypoint.
#         #   - In the first layer, the H is equal to the (num_keypoints, 2*num_samples) input matrix X.

#         # W (weight):  (in_features, out_features)
#         #   - We can change out_features to control the number of output features.

#         # HW (batch_size, num_keypoints, out_features): i.e. we multiply each X_i in the batch by W.
#         #       - Essentially, HW is a matrix that stores a set number of features (determined by out_features) for
#         #         each keypoint.
#         support = input @ self.weight

#         # A (adjacency matrix): (num_keypoints, num_keypoints)

#         # AHW (output): (batch_size, num_keypoints, out_features)
#         #   - For each keypoint K in AHW, we calculate its corresponding features f as follows:
#         #       - Compute the weighted sum between the edge weights from K to every other keypoint and 
#         #         the value of f for each keypoint.
#         output = self.att @ support  # g = A*HW

#         return output + self.bias

#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.in_features) + ' -> ' \
#                + str(self.out_features) + ')'


# class GCN_2Layer(nn.Module):
#     """2-Layer GCN model specified in https://arxiv.org/abs/1811.05320"""

#     def __init__(self, in_features, p_dropout):
#         super(GCN_2Layer, self).__init__()
#         self.gc1 = GraphConvolution(in_features, in_features)
#         self.bn1 = nn.BatchNorm1d(55 * 100)

#         self.gc2 = GraphConvolution(in_features, in_features)
#         self.bn2 = nn.BatchNorm1d(55 * 100)

#         self.do = nn.Dropout(p_dropout)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, x):
#         y = self.gc1(x)
#         b, n, k, f = y.shape
#         y = self.bn1(y.view(b, -1)).view(b, n, k, f)
#         y = self.relu(y)
#         y = self.do(y)

#         y = self.gc2(y)
#         b, n, k, f = y.shape
#         y = self.bn2(y.view(b, -1)).view(b, n, k, f)
#         y = self.sigmoid(y)
#         y = self.do(y)

#         return y

class TGCN_GRU(nn.Module):
    def __init__(self, input_size, p_dropout):
        super(TGCN_GRU, self).__init__()
        self.gcn = GC_Block(in_features=input_size, p_dropout=p_dropout)
        self.gru = nn.GRU(input_size=110,
                          hidden_size=110,
                          num_layers=1,
                          bias=True,
                          batch_first=True)
        self.do = nn.Dropout1d(0.3)
        self.weight = Parameter(torch.FloatTensor(110, 100))
        self.bias = Parameter(torch.FloatTensor(100)) 
    
    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv1, stdv1)
        self.bias.data.uniform_(-stdv1, stdv1)

    def forward(self, x):
        y = self.gcn(x)
        y = torch.stack(torch.chunk(y, 50, 2))
        y = torch.permute(y, (1, 0, 2, 3))
        y = torch.flatten(y, start_dim=2)
        _, y = self.gru(y)
        y = y.squeeze() @ self.weight
        y = y + self.bias

        return y
