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
# explicit relative import
# from ..utils import MixedLinear, MixedDropout

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""

class Pointer():
    def __init__(self, value):
        self.value = value
        
class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    
    Set of modules for aggregating embeddings of neighbors.

    """
    def __init__(self, features, prob_matrix, monstor_style=False, gcn=False, cuda=False,): 
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.prob_matrix = prob_matrix
        self.monstor_style = monstor_style
        self.cuda = cuda
        self.gcn = gcn
        
    def forward(self, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        # import ipdb; ipdb.set_trace()
        _set = set
        if num_sample is not None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else _set(to_neigh) for to_neigh in to_neighs]
        else:
            samp_neighs = _set(to_neighs)

        if self.gcn:
            # import ipdb; ipdb.set_trace()
            samp_neighs = [samp_neigh | set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}

        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]  # node -> index in unique_nodes_list
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))] # mask mat's which row     

        # self.monstor_style = False
        if self.monstor_style: # MONSTOR aggregates node i's neighbor j's features with * p_ij
            c_indices = [unique_nodes_list[i] for i in column_indices] # index in unique_nodes_list -> node
            mask_data = np.zeros((len(samp_neighs), len(unique_nodes)))
            mask_data[row_indices, column_indices] = self.prob_matrix.value[c_indices, row_indices]
            mask = Variable( torch.FloatTensor(mask_data) )
            # mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        else:
            mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
            mask[row_indices, column_indices] = 1
            num_neigh = mask.sum(1, keepdim=True)
            mask = mask.div(num_neigh)
        

        if self.cuda:
            mask = mask.cuda()
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        
        # ipdb.set_trace()
        to_feats = mask.mm(embed_matrix) # mean aggregation
        # ipdb.set_trace()

        return to_feats

class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, 
            features, 
            feature_dim, 
            embed_dim, 
            adj_lists, 
            aggregator,
            num_sample=10, 
            gcn=False, 
            cuda=False): 
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        self.weight = nn.Parameter(
                torch.FloatTensor(self.feat_dim*2 if not self.gcn else self.feat_dim, self.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes: torch.LongTensor):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes, tensor
        """
        # import ipdb; ipdb.set_trace()
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists.value[int(node)] for node in nodes], self.num_sample)
        if not self.gcn: # GraphSAGE style
            if self.cuda:
                # ipdb.set_trace()
                self_feats = self.features(nodes).to('cuda')
                # ipdb.set_trace()
            else:
                self_feats = self.features(nodes)
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else: # GCN style
            combined = neigh_feats
        
        combined = F.relu( torch.mm(combined, self.weight))
        # print(combined.shape)
        return combined

class SupervisedGraphSage(nn.Module):
    """
    GraphSAGE module
    for regression, so the loss is mean square error.
    """

    def __init__(self, 
                feature_mat: np.array, 
                prob_matrix,
                adj_lists: np.array, 
                device,
                num_classes=1,
                monstor_flag=False,
                gcn=False):
        """
        args:
            feature_mat: N*dim matrix, feature matrix after feature propagation using personalized page rank vector.
            adj_lists: N*N matrix.

        """
        super(SupervisedGraphSage, self).__init__()
        assert isinstance(feature_mat, np.ndarray), "feature matrix must be a np array, not sparse array"
        # embedding layer
        self.monstor_flag = monstor_flag
        self.N = feature_mat.shape[0]
        self.dim = feature_mat.shape[1]
        self.prob_matrix = Pointer(prob_matrix)
        self.adj_lists = Pointer(adj_lists)

        self.features = nn.Embedding(self.N, self.dim)
        self.features.weight = nn.Parameter(torch.FloatTensor(feature_mat), requires_grad=False)

        if device =='cuda':
            cuda = True
        else: cuda = False
        # 2 graphsage layers
        self.agg1 = MeanAggregator(self.features, self.prob_matrix, monstor_style=self.monstor_flag, gcn=gcn, cuda=cuda) # module instance
        self.enc1 = Encoder(self.features, self.dim, 64, self.adj_lists, self.agg1, gcn=gcn, cuda=cuda) # modlue instance
        self.agg2 = MeanAggregator(self.enc1, self.prob_matrix, monstor_style=self.monstor_flag, gcn=gcn, cuda=cuda) # module instance
        self.enc2 = Encoder(self.enc1, self.enc1.embed_dim, 64, self.adj_lists, self.agg2, gcn=gcn, cuda=cuda) # module instance

        self.enc1.num_samples = 10
        self.enc2.num_samples = 10

        if self.monstor_flag: # monstor uses 3 layers
            self.enc = self.enc2       
        else:
            self.enc = self.enc2

        # final linear weight
        self.weight = nn.Parameter(torch.FloatTensor(self.enc.embed_dim, num_classes))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        # import ipdb; ipdb.set_trace()
        embeds = self.enc(nodes) # self.enc, is a module instance.
        scores = torch.mm(embeds, self.weight)
        predictions = torch.sigmoid(scores)

        if self.monstor_flag:
            device = next(self.features.parameters()).device
            predictions = predictions.flatten()
            predictions = self.features(nodes)[:, -1] + predictions
            prob_matrix = torch.FloatTensor(self.prob_matrix.value.toarray().T).to(device)
            g = self.features(nodes)[:, -1] - self.features(nodes)[:, -2]
            # ipdb.set_trace()
            upp_bound = self.features(nodes)[:, -1] +  prob_matrix @ g
            predictions = torch.min(predictions, upp_bound)
        return predictions




if __name__ == "__main__":
    pass
