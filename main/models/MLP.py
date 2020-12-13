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

class MLPTransform(nn.Module):
    def __init__(self, 
                input_dim,
                hiddenunits: List[int],
                num_classes,
                bias=True,
                drop_prob=0.5):
        super(MLPTransform, self).__init__()
        # Here features is just a placeholder, each time before forward, we will substutute the embedding layer with desired node feature matrix
        # and when saving model params, we will first pop self.features.weight
        self.features = None

        fcs = [nn.Linear(input_dim, hiddenunits[0], bias=bias)]
        for i in range(1, len(hiddenunits)):
            fcs.append(nn.Linear(hiddenunits[i-1], hiddenunits[i]))
        fcs.append(nn.Linear(hiddenunits[-1], num_classes))

        self.fcs = nn.ModuleList(fcs)

        if drop_prob is 0:
            self.dropout = lambda x: x
        else:
            self.dropout = nn.Dropout(drop_prob)
        self.act_fn = nn.ReLU()

    def forward(self, nodes: torch.LongTensor):
        # ipdb.set_trace()
        layer_inner = self.act_fn(self.fcs[0](self.dropout(self.features(nodes))))
        for fc in self.fcs[1:-1]:
            layer_inner = self.act_fn(fc(layer_inner))
        res = torch.sigmoid( self.fcs[-1](self.dropout(layer_inner)) )
        return res


