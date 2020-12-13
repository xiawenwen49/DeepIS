import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import ipdb

import numpy as np
import time
import random
from sklearn.metrics import f1_score, accuracy_score
from collections import defaultdict
from typing import List


class GATLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, alpha, bias=True):
        super(GATLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.alpha = alpha
        self.bias = bias

        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.a = nn.Parameter(torch.FloatTensor(2*output_dim, 1))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
    
    def forward(self, features, edge_lists):
        x = F.dropout(features, self.dropout, training=self.training)
        h = torch.mm(x, self.weight)

        source, target = edge_lists
        device = source.device
        atten_input = torch.cat([h[source], h[target]], dim=1)
        e = F.leaky_relu( torch.mm(atten_input, self.a), negative_slope=self.alpha) # attention vector
        N = features.shape[0]
        # import ipdb; ipdb.set_trace()
        attention = -1e10*torch.ones((N, N), requires_grad=True, device=device) # attention matrix
        
        attention[source, target] = e[:, 0] # feed the attention matrix

        attention = torch.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h = F.dropout(h, self.dropout, training=self.training)
        h_prime = torch.mm(attention, h)
        if self.bias is not None:
            h_prime = h_prime + self.bias
        
        return h_prime

        
class GAT(nn.Module):
    """For both regression and classification."""
    def __init__(self, input_dim, output_dim, adj_lists, nhid=8, nhead=4, nhead_out=1, alpha=0.2, dropout=0.5):
        super(GAT, self).__init__()    

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.features = nn.Embedding(1, input_dim) # placeholder
        self.features.weight = nn.Parameter(torch.zeros(1, input_dim), requires_grad=False)
        self._adj_lists = adj_lists
        self.atten = [GATLayer(input_dim, nhid, dropout, alpha) for _ in range(nhead)]
        self.atten_out = [GATLayer(nhead*nhid, output_dim, dropout, alpha) for _ in range(nhead_out)]

        for i, attention in enumerate(self.atten):
            self.add_module('attentioin_{}'.format(i), attention)
        for i, attention in enumerate(self.atten_out):
            self.add_module('out_att{}'.format(i), attention)
        self.reset_parameters()

    def reset_parameters(self):
        for att in self.atten:
            att.reset_parameters()
        for att in self.atten_out:
            att.reset_parameters()
    
    def update_adj_lists(self, adj_lists):
        self._adj_lists = adj_lists

    def forward(self, nodes: torch.LongTensor):
        device = self.features.weight.device
        source = []
        target = []
        for i, neighs in enumerate(self._adj_lists):
            for j in neighs:
                source.append(i)
                target.append(j)
        edge_lists = [torch.LongTensor(source).to(device), torch.LongTensor(target).to(device)]
        x = torch.cat([att(self.features.weight, edge_lists) for att in self.atten], dim=1)
        x = F.elu(x)
        x = torch.sum( torch.stack([att(x, edge_lists) for att in self.atten_out]), dim=0) / len(self.atten_out)

        if self.output_dim == 1:
            return torch.sigmoid(x)[nodes]
        else:
            return torch.softmax(x, dim=1)[nodes]


