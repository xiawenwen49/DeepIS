import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import ipdb
import scipy.sparse as sp
import numpy as np
import time
import random
from sklearn.metrics import f1_score, accuracy_score
from collections import defaultdict
from typing import List

class SGC(nn.Module):
    def __init__(self, input_dim, num_class, prob_matrix, niter):
        super().__init__()
        assert niter > 0, 'Invalid niter'
        if sp.isspmatrix(prob_matrix):
            prob_matrix = prob_matrix.toarray()

        self.input_dim = input_dim
        self.num_class = num_class
        P = np.linalg.matrix_power(prob_matrix.T, niter)
        self.niter = niter
        self.register_buffer('P', torch.FloatTensor(P.T))
        self.weight = nn.Parameter(torch.ones((input_dim*2, num_class)), requires_grad=True)
        self.features = None
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, nodes):
        layer_inner = self.features(nodes)
        layer_inner_ = self.P @ layer_inner
        layer_inner = torch.cat([layer_inner, layer_inner_], axis=1)
        layer_inner = layer_inner @ self.weight
        if self.num_class == 1:
            preds = torch.sigmoid(layer_inner)
        else:
            preds = torch.log_softmax(preds, dim=-1)
        return preds


