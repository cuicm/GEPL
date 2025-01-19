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
    Graph Attention Network Layer (Batch support: input shape (B, N, F))
    """
    def __init__(self, input_dim, output_dim, num_heads=1, alpha=0.2, concat=True):
        super(GraphAttention, self).__init__()
        self.in_features = input_dim
        self.out_features = output_dim
        self.alpha = alpha

        # Multi-head weight matrices (num_heads, input_dim, output_dim)
        self.W = nn.Parameter(torch.FloatTensor(num_heads, input_dim, output_dim))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Attention mechanism weights (num_heads, 2*output_dim, 1)
        self.a = nn.Parameter(torch.FloatTensor(num_heads, 2 * output_dim, 1))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(self.alpha)

        self.num_heads = num_heads
        self.concat = concat

    def forward(self, x, adj):

        B, N, _ = x.size()  # Batch size, number of nodes, input feature dimension

        # Add self-loops to adjacency matrices
        adj = adj + torch.eye(N, device=adj.device).unsqueeze(0).repeat(B, 1, 1)

        # Linear transformation: (B, N, F) -> (B, num_heads, N, output_dim)
        x_transformed = torch.einsum('bni,hio->bhno', x, self.W)

        # Compute attention scores
        # Repeat features for pairwise concatenation: (B, num_heads, N, N, 2*output_dim)
        f_repeat = x_transformed.unsqueeze(3).repeat(1, 1, 1, N, 1)
        f_repeat_interleave = x_transformed.unsqueeze(2).repeat(1, 1, N, 1, 1)
        all_features = torch.cat([f_repeat, f_repeat_interleave], dim=-1)  # (B, num_heads, N, N, 2*output_dim)

        # Apply attention mechanism: (B, num_heads, N, N, 2*output_dim) -> (B, num_heads, N, N)
        attention_scores = self.leaky_relu(torch.matmul(all_features, self.a).squeeze(-1))  # Remove last dimension

        # Mask attention scores with adjacency matrix
        zero_vec = -9e15 * torch.ones_like(attention_scores)
        attention_scores = torch.where(adj.unsqueeze(1) > 0, attention_scores, zero_vec)

        # Normalize attention scores with softmax
        attention_scores_normalized = F.softmax(attention_scores, dim=-1)

        # Compute attention-weighted features: (B, num_heads, N, output_dim)
        h_prime = torch.einsum('bhij,bhjd->bhid', attention_scores_normalized, x_transformed)

        if self.concat:
            h_prime = h_prime.permute(0, 2, 1, 3).contiguous().view(B, N, -1)  # (B, N, num_heads * output_dim)
        else:
            h_prime = h_prime.mean(dim=1)  # (B, N, output_dim)

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


class GraphSAGE(nn.Module):
    """
    GraphSAGE Layer
    """
    def __init__(self, input_dim, output_dim, aggregator="mean"):
        """
        Args:
            input_dim: Input feature dimension.
            output_dim: Output feature dimension.
            aggregator: Aggregation method ("mean", "max", or "pool").
            activation: Activation function (default: ReLU).
            dropout: Dropout rate (default: 0.0).
        """
        super(GraphSAGE, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.aggregator = aggregator

        # Linear layers for transforming node and neighbor features
        self.linear_self = nn.Linear(input_dim, output_dim)  # For the node's own features
        self.linear_neighbors = nn.Linear(input_dim, output_dim)  # For aggregated neighbors' features

        # Pooling layer for "pool" aggregator
        if aggregator == "pool":
            self.pooling_layer = nn.Linear(input_dim, input_dim)

    def forward(self, x, adj):
        if len(x.shape)<3:
            x=x.unsqueeze(0)
        if len(adj.shape)<3:
            adj=adj.unsqueeze(0)
        """
        Args:
            x: Input node features of shape (B, N, F).
            adj: Adjacency matrix of shape (B, N, N).
        Returns:
            Output node features of shape (B, N, output_dim).
        """
        B, N, F = x.shape

        # Neighbor aggregation based on the chosen aggregator
        if self.aggregator == "mean":
            # Mean aggregation: Aggregate neighbor features as the mean of neighbors
            neighbor_features = torch.bmm(adj, x) / (adj.sum(dim=-1, keepdim=True) + 1e-10)
        elif self.aggregator == "max":
            # Max aggregation: Max-pool neighbor features
            mask = adj.unsqueeze(-1)  # Shape: (B, N, N, 1)
            neighbor_features = x.unsqueeze(1).repeat(1, N, 1, 1)  # Shape: (B, N, N, F)
            neighbor_features = neighbor_features.masked_fill(mask == 0, float("-inf"))
            neighbor_features, _ = neighbor_features.max(dim=2)  # Shape: (B, N, F)
        elif self.aggregator == "pool":
            # Pool aggregation: Apply non-linear transformation before aggregation
            pooled = F.relu(self.pooling_layer(x))  # Shape: (B, N, F)
            neighbor_features = torch.bmm(adj, pooled)  # Shape: (B, N, F)
        else:
            raise ValueError(f"Unsupported aggregator type: {self.aggregator}")

        # Transform the node's own features
        self_features = self.linear_self(x)  # Shape: (B, N, output_dim)

        # Transform the aggregated neighbor features
        neighbor_features = self.linear_neighbors(neighbor_features)  # Shape: (B, N, output_dim)

        # Combine the node's own features and the aggregated neighbor features
        out = self_features + neighbor_features  # Shape: (B, N, output_dim)

        return out


class Brain_GCN(Module):
    '''
    Encoder
    '''
    def __init__(self,input_dim, hidden_dim, num_layers=5, drop_ratio=0, graph_pooling="add", gtype='gcn'):
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
                if gtype == 'gcn':
                    self.convs.append(GraphConvolution(input_dim, hidden_dim))
                elif gtype == 'gin':
                    self.convs.append(GraphIsomorphism(input_dim, hidden_dim))
                elif gtype == 'gat':
                    self.convs.append(GraphAttention(input_dim, hidden_dim))
                elif gtype == 'sage':
                    self.convs.append(GraphSAGE(input_dim, hidden_dim))
            else:
                if gtype == 'gcn':
                    self.convs.append(GraphConvolution(hidden_dim, hidden_dim))
                elif gtype == 'gin':
                    self.convs.append(GraphIsomorphism(hidden_dim, hidden_dim))
                elif gtype == 'gat':
                    self.convs.append(GraphAttention(hidden_dim, hidden_dim))
                elif gtype == 'sage':
                    self.convs.append(GraphSAGE(hidden_dim, hidden_dim))
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
        if len(x.shape)<3: x=x.unsqueeze(0)
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




