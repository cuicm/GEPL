import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import os
import numpy as np
import random

def global_add_pool(x: Tensor, adj: Tensor) -> Tensor:
    return x.sum(dim=1)  

def global_mean_pool(x: Tensor, adj: Tensor) -> Tensor:
    num_nodes = x.shape[1]
    return x.sum(dim=1) / num_nodes  

def global_max_pool(x: Tensor, adj: Tensor) -> Tensor:
    return x.max(dim=1)[0] 

def global_concat_pool(x: Tensor, adj: Tensor) -> Tensor:
    return x.view(x.size(0), -1)

class GlobalAttentionPool(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super(GlobalAttentionPool, self).__init__()
        self.attention = nn.Linear(in_features, hidden_dim)
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, x, adj):

        h = torch.tanh(self.attention(x)) 
        scores = self.proj(h)  
        scores = scores.squeeze(-1)  
        attention_weights = F.softmax(scores, dim=1)  
        attention_weights = attention_weights.unsqueeze(-1)  
        
        x = x * attention_weights  
        readout = torch.sum(x, dim=1) 
        
        return readout


def caculate_adj_matrix(x):
    adj_matrix = abs(np.corrcoef(x))
    return adj_matrix

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ChannelDropout(object):
    def __init__(self, p=0.5):
        super(ChannelDropout, self).__init__()
        self.p = p

    def __call__(self, x, adj):
        N = x.shape[0]
        keep_prob = 1 - self.p
        drop_mask = torch.rand(N) < keep_prob
        x = x[drop_mask]
        adj = adj[drop_mask][:, drop_mask]
        return x,adj

class ConnectionDropout(object):
    def __init__(self, p=0.5):
        super(ConnectionDropout, self).__init__()
        self.p = p

    def __call__(self, x, adj):
        dropout_mask = (torch.rand_like(adj) > self.p).float()
        adj = adj * dropout_mask
        return x,adj

class FeatureMask(object):
    def __init__(self, p=0.5) -> None:
        super(FeatureMask, self).__init__()
        self.p = p
    
    def __call__(self, x, adj):
        mask = (torch.rand_like(x) > self.p).float()
        x = x * mask
        return x, adj



