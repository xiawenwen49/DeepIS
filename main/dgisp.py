import torch
import torch.nn as nn
import torch.functional as F
import numpy as np 
import ipdb
import scipy.sparse as sp

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, preds, seed_idx, idx):
        return preds[idx]

class DiffusionPropagate(nn.Module):
    def __init__(self, prob_matrix, iter_n):
        super(DiffusionPropagate, self).__init__()
        self.iter_n = iter_n # 迭代次数
        if sp.isspmatrix(prob_matrix):
            prob_matrix = prob_matrix.toarray()
        
        self.register_buffer('prob_matrix', torch.FloatTensor(prob_matrix))
    
    def forward(self, preds, seed_idx, idx):
        # import ipdb; ipdb.set_trace()
        device = preds.device
        for i in range(self.iter_n):
            P2 = self.prob_matrix.T * preds.view((1, -1)).expand(self.prob_matrix.shape)
            P3 = torch.ones(self.prob_matrix.shape).to(device) - P2
            preds = torch.ones((self.prob_matrix.shape[0], )).to(device) - torch.prod(P3, dim=1)
            preds[seed_idx] = 1

        return preds[idx]

class DGISP(nn.Module):
    """
    Essentially it's a node regression task.
    """
    def __init__(self, gnn_model: nn.Module, propagate: nn.Module):
        super(DGISP, self).__init__()
        self.gnn_model = gnn_model 
        self.propagate = propagate

        self.reg_params = list( filter( lambda x: x.requires_grad, self.gnn_model.parameters()) )
        
    def forward(self, idx: torch.LongTensor):
        """
        index
            node indexs that we want to get
        actually we will predict values of all nodes; the index just indicates which values to be fetched
        """
        device = next(self.gnn_model.parameters()).device
        total_node_nums = self.gnn_model.features.weight.shape[0]
        total_nodes = torch.LongTensor( np.arange(total_node_nums) ).to(device)

        seed_idx = torch.LongTensor( np.argwhere( self.gnn_model.features.weight.detach().cpu().numpy()[:, 0] == 1) ).to(device)
        
        predictions = self.gnn_model(total_nodes) # predict all, for prediction propagation
        
        predictions = self.propagate(predictions, seed_idx, idx) # then select

        return predictions.flatten()
         

    # @staticmethod
    def loss(self, predictions, labels, λ, γ):

        # L1 = self.mse_loss(predictions, labels)
        L1 = torch.sum( torch.abs(predictions - labels) ) / len(labels) # node-level error
        L2 = torch.abs(torch.sum(predictions) - torch.sum(labels))/(torch.sum(labels) + 1e-5) # influence spread error
#         L2 = torch.abs(torch.sum(predictions) - torch.sum(labels))/(len(labels))
        Reg = sum(torch.sum(param ** 2) for param in self.reg_params)
        Loss = L1 + λ*L2 + γ * Reg
        # Loss = L1 + λ*L2
        return Loss