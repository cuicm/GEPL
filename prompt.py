import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class GraphPrompt(nn.Module):
    def __init__(self, in_channels: int):
        super(GraphPrompt, self).__init__()
        self.global_emb = nn.Parameter(torch.Tensor(1, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.global_emb)

    def add(self, x: Tensor):
        return x * self.global_emb


class GPFplusAtt(nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(GPFplusAtt, self).__init__()
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.p_list)
        self.a.reset_parameters()

    def add(self, x: Tensor):
        score = self.a(x)
        weight = F.softmax(score, dim=2)
        p = torch.matmul(weight,self.p_list)
        return x * p


class EdgeMask(nn.Module):
    def __init__(self, num_nodes):
        super(EdgeMask, self).__init__()
        self.num_nodes = num_nodes
        self.prompt_adj = nn.Parameter(torch.FloatTensor(self.num_nodes, self.num_nodes), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.prompt_adj)

    def mask(self,adj,self_loop=True,device='cuda'):
        adj_p = F.relu(adj * (self.prompt_adj + self.prompt_adj.transpose(1, 0)))
        if self_loop:
            adj_p = adj_p + torch.eye(self.num_nodes).to(device)
        return adj_p

    def print_adj(self):
        adj_sum = self.prompt_adj + self.prompt_adj.transpose(1, 0)
        for i in range(adj_sum.shape[0]):
            for j in range(adj_sum.shape[1]):
                print(f"{adj_sum[i][j].item():.4f}", end=' ')
            print()







   