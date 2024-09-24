import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
from utils import global_add_pool,global_max_pool,global_mean_pool,global_concat_pool,GlobalAttentionPool

def normalization(adj):
    flag = False
    if adj.dim() == 2:
        adj = adj.unsqueeze(0)
        flag = True

    rowsum = torch.sum(adj, dim=-1)
    mask = torch.zeros_like(rowsum)
    mask[rowsum == 0] = 1
    rowsum += mask
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
    adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

    if flag:
        adj = adj.squeeze(0)

    return adj


class GraphConvolution(Module):
    """
    Graph Convolutional Network Layer
    """
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weights = Parameter(torch.FloatTensor(input_dim, output_dim))
        torch.nn.init.xavier_uniform_(self.weights, gain=1.414)
        if self.use_bias:
            self.bias = Parameter(torch.zeros((1, 1, output_dim), dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weights, gain=1.414)
    
    def forward(self, features, adjacency):
        output = torch.matmul(features, self.weights)
        output = torch.matmul(adjacency, output)
        if self.use_bias:
            output = output + self.bias
        return output
    

class GraphAttention(nn.Module):
    """
    Graph Attention Network Layer
    """
    def __init__(self, input_dim, output_dim, num_heads=1, use_bias=True):
        super(GraphAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.use_bias = use_bias

        self.weights = Parameter(torch.FloatTensor(input_dim, output_dim * num_heads))
        self.attention = Parameter(torch.FloatTensor(1, num_heads, 2 * output_dim))  # For concatenating the self and neighbor features

        torch.nn.init.xavier_uniform_(self.weights, gain=1.414)
        torch.nn.init.xavier_uniform_(self.attention, gain=1.414)

        if self.use_bias:
            self.bias = Parameter(torch.zeros(output_dim * num_heads, dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weights, gain=1.414)
        torch.nn.init.xavier_uniform_(self.attention, gain=1.414)

    def forward(self, features, adjacency):
        N = features.size()[0]
        h = torch.matmul(features, self.weights).view(N, self.num_heads, self.output_dim)
        
        a_input = torch.cat([h.repeat(1, 1, N).view(N * N, self.num_heads, -1), 
                             h.repeat(1, N, 1)], dim=2).view(N, N, self.num_heads, 2 * self.output_dim)
        e = F.leaky_relu(torch.einsum('ijhd,hd->ijh', a_input, self.attention), negative_slope=0.2)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adjacency > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)

        h_prime = torch.einsum('ijh,jhd->ihd', attention, h).view(N, self.num_heads * self.output_dim)

        if self.use_bias:
            h_prime = h_prime + self.bias

        return h_prime

class GraphIsomorphism(nn.Module):
    """
    Graph Isomorphism Network Layer
    """
    def __init__(self, input_dim, output_dim, epsilon=0, use_bias=True):
        super(GraphIsomorphism, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epsilon = nn.Parameter(torch.Tensor([epsilon]))
        self.use_bias = use_bias

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 2 * output_dim),
            nn.ReLU(),
            nn.Linear(2 * output_dim, output_dim)
        )

        if self.use_bias:
            self.bias = Parameter(torch.zeros(output_dim, dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight, gain=1.414)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

    def forward(self, features, adjacency):
        h = (1 + self.epsilon) * features + torch.matmul(adjacency, features)
        output = self.mlp(h)

        if self.use_bias:
            output = output + self.bias

        return output

class Brain_GCN(Module):
    '''
    Encoder
    '''
    def __init__(self,input_dim, hidden_dim, num_layers=5, drop_ratio=0, graph_pooling="add"):
        super(Brain_GCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.graph_pooling = graph_pooling

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(self.num_layers):
            if i == 0:
                self.convs.append(GraphConvolution(input_dim, hidden_dim))
            else:
                self.convs.append(GraphConvolution(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        if self.graph_pooling == "add":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == 'concat':
            self.pool = global_concat_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttentionPool(hidden_dim,int(hidden_dim/2))

    def forward(self, x, adj):
        adj = normalization(adj)
        for i in range(self.num_layers):
            x = self.convs[i](x, adj)
            x = F.relu(x) 
            x = F.dropout(x, self.drop_ratio, training=self.training)

        g = self.pool(x, adj)
        return x,g

class Encoder(nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, adj):
        aug1, aug2 = self.augmentor
        x1, adj1 = aug1(x, adj)
        x2, adj2 = aug2(x, adj)
        x, g = self.encoder(x, adj)
        x1, g1 = self.encoder(x1, adj1)
        x2, g2 = self.encoder(x2, adj2)
        return x, g, x1, x2, g1, g2

class Projection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Projection, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EEG_model(nn.Module):
    def __init__(self, encoder, input_dim, hidden_dim, classes, drop_out=0):
        super(EEG_model, self).__init__()
        self.encoder = encoder
        self.MLP = nn.Linear(hidden_dim, classes)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.classes = classes
        self.drop_out = drop_out

    def forward(self,x,adj,fp=None,sp=None):
        if fp != None:
            x = fp.add(x)
        if sp != None:
            adj = sp.mask(adj)
        x, g = self.encoder(x, adj)
        out = F.dropout(g, p=self.drop_out, training=self.training)
        out = self.MLP(out)
        return out


