import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, input):
        input_coal = input.coalesce()
        drop_val = F.dropout(input_coal._values(), self.p, self.training)
        return torch.sparse.FloatTensor(input_coal._indices(), drop_val, input.shape)


class MixedDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dense_dropout = nn.Dropout(p)
        self.sparse_dropout = SparseDropout(p)

    def forward(self, input):
        if input.is_sparse:
            return self.sparse_dropout(input)
        else:
            return self.dense_dropout(input)


class MixedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Our fan_in is interpreted by PyTorch as fan_out (swapped dimensions)
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.bias is None:
            if input.is_sparse:
                res = torch.sparse.mm(input, self.weight)
            else:
                res = input.matmul(self.weight)
        else:
            if input.is_sparse:
                res = torch.sparse.addmm(self.bias.expand(input.shape[0], -1), input, self.weight)
            else:
                res = torch.addmm(self.bias, input, self.weight)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
                self.in_features, self.out_features, self.bias is not None)


def sparse_matrix_to_torch(X):
    coo = X.tocoo()
    indices = np.array([coo.row, coo.col])
    return torch.sparse.FloatTensor( # torch.sparse
            torch.LongTensor(indices),
            torch.FloatTensor(coo.data),
            coo.shape)


def matrix_to_torch(X):
    if sp.issparse(X):
        return sparse_matrix_to_torch(X)
    else:
        return torch.FloatTensor(X)


def to_torch(X):
    if sp.issparse(X):
        X = to_nparray(X)
    return torch.FloatTensor(X)

def to_nparray(X):
    if sp.isspmatrix(X):
        return X.toarray()
    else: return X

def sp2adj_lists(X):
    assert sp.isspmatrix(X), 'X should be sp.sparse'
    adj_lists = []
    if sp.isspmatrix(X):
        for i in range(X.shape[0]):
            neighs = list( X[i,:].nonzero()[1] )
            adj_lists.append(neighs)
        return adj_lists
    else:
        return None


def load_dataset(dataset, data_dir='data'):
    from pathlib import Path
    import pickle
    import sys

    sys.path.append('data') # for pickle.load

    data_dir = Path(data_dir)
    suffix = '_25c.SG'
    graph_name = dataset + suffix
    path_to_file = data_dir / graph_name
    with open(path_to_file, 'rb') as f:
        graph = pickle.load(f)
    return graph



def load_latest_ckpt(model_name, dataset, ckpt_dir='./checkpoints'): # get最新的checkpoint
    from pathlib import Path
    ckpt_dir = Path(ckpt_dir)
    ckpt_files = []
    for p in ckpt_dir.iterdir():
        if model_name in str(p) and dataset in str(p):
            ckpt_files.append(str(p)) 
    if len(ckpt_files) > 0:
        ckpt_file = sorted(ckpt_files, key=lambda x: x[-22:])[-1] 
    else: raise FileNotFoundError
    print('checkpoint file:', ckpt_file)
    import torch
    state_dict = torch.load(ckpt_file)
    return state_dict  




    