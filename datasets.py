import torch
from torch.utils.data import Dataset, DataLoader
from read_raw_files import read_raw_tinnus,read_raw_deap,read_raw_shl,read_raw_alzh,read_raw_TE
from preprocess import preprocess_eeg
from utils import caculate_adj_matrix

def trans_adj(adj_matrix, k):
    num_nodes = adj_matrix.shape[0]
    for i in range(num_nodes):
        neighbors = [(j, adj_matrix[i, j].item()) for j in range(num_nodes) if i != j]
        neighbors.sort(key=lambda x: x[1], reverse=True)
        top_k_neighbors = neighbors[:k]
        
        new_row = torch.zeros_like(adj_matrix[i])
        new_row[i] = adj_matrix[i, i]

        for j, weight in top_k_neighbors:
            new_row[j] = weight
        adj_matrix[i] = new_row
    adj_matrix = (adj_matrix + adj_matrix.t()) / 2.0
    return adj_matrix


class Pretrain_Dataset(Dataset):
    def __init__(self,t_dir,d_dir,sample_rate,window_length):
        self.sample_rate = sample_rate
        self.window_length = window_length
        self.data=[]

        signals = read_raw_tinnus(t_dir)
        for i in range(len(signals)):
            signals[i] = preprocess_eeg(signals[i],128,sample_rate,window_length)
            node_features = torch.tensor(signals[i])
            adj_matrix = torch.tensor(caculate_adj_matrix(signals[i]))
            adj_matrix = trans_adj(adj_matrix, 2)
            self.data.append((node_features, adj_matrix))
        
        signals = read_raw_deap(d_dir)
        for i in range(len(signals)):
            signals[i] = preprocess_eeg(signals[i],128,sample_rate,window_length)
            node_features = torch.tensor(signals[i])
            adj_matrix = torch.tensor(caculate_adj_matrix(signals[i]))
            adj_matrix = trans_adj(adj_matrix, 1)
            self.data.append((node_features, adj_matrix))
        self.length = len(self.data)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.data[idx]

class Tinnus_Dataset(Dataset):
    def __init__(self,raw_dir,sample_rate,window_length):
        self.sample_rate = sample_rate
        self.window_length = window_length
        self.data=[]

        signals = read_raw_tinnus(raw_dir)
        for i in range(len(signals)):
            signals[i] = preprocess_eeg(signals[i],128,sample_rate,window_length)
            node_features = torch.tensor(signals[i])
            adj_matrix = torch.tensor(caculate_adj_matrix(signals[i]))
            adj_matrix = trans_adj(adj_matrix, 2)
            self.data.append((node_features, adj_matrix))
        self.length = len(self.data)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.data[idx]

class Deap_Dataset(Dataset):
    def __init__(self,raw_dir,sample_rate,window_length):
        self.sample_rate = sample_rate
        self.window_length = window_length
        self.data=[]

        signals = read_raw_deap(raw_dir)
        for i in range(len(signals)):
            signals[i] = preprocess_eeg(signals[i],128,sample_rate,window_length)
            node_features = torch.tensor(signals[i])
            adj_matrix = torch.tensor(caculate_adj_matrix(signals[i]))
            adj_matrix = trans_adj(adj_matrix, 1)
            self.data.append((node_features, adj_matrix))
        self.length = len(self.data)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.data[idx]
        

class SHL_Dataset(Dataset):
    def __init__(self,raw_dir,sample_rate,window_length):
        self.sample_rate = sample_rate
        self.window_length = window_length
        self.data = []
        self.labels = []

        signals,labels = read_raw_shl(raw_dir)
        self.labels = [torch.tensor(label) for label in labels]
        for i in range(len(signals)):
            signals[i] = preprocess_eeg(signals[i],128,sample_rate,window_length)
            node_features = torch.tensor(signals[i])
            adj_matrix = torch.tensor(caculate_adj_matrix(signals[i]))
            self.data.append((node_features, adj_matrix))
        self.length = len(self.data)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class Alzh_Dataset(Dataset):
    def __init__(self,raw_dir,sample_rate,window_length):
        self.sample_rate = sample_rate
        self.window_length = window_length
        self.data = []
        self.labels = []

        signals,labels = read_raw_alzh(raw_dir)
        self.labels = [torch.tensor(label) for label in labels]
        for i in range(len(signals)):
            signals[i] = preprocess_eeg(signals[i],500,sample_rate,window_length)
            node_features = torch.tensor(signals[i])
            adj_matrix = torch.tensor(caculate_adj_matrix(signals[i]))
            self.data.append((node_features, adj_matrix))
        self.length = len(self.data)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class TE_Dataset(Dataset):
    def __init__(self,raw_dir,sample_rate,window_length):
        self.sample_rate = sample_rate
        self.window_length = window_length
        self.data = []
        self.labels = []

        signals,labels = read_raw_TE(raw_dir)
        self.labels = [torch.tensor(label) for label in labels]
        for i in range(len(signals)):
            signals[i] = preprocess_eeg(signals[i],500,sample_rate,window_length)
            node_features = torch.tensor(signals[i])
            adj_matrix = torch.tensor(caculate_adj_matrix(signals[i]))
            self.data.append((node_features, adj_matrix))
        self.length = len(self.data)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def collate_fn(batch):
    node_features_batch = [item[0] for item in batch]
    adj_matrix_batch = [item[1] for item in batch]
    return node_features_batch, adj_matrix_batch

def collate_fn_ft(batch):
    data_batch, labels_batch = zip(*batch)
    node_features_batch, adj_matrix_batch = zip(*data_batch)
    node_features_batch = torch.stack(node_features_batch)
    adj_matrix_batch = torch.stack(adj_matrix_batch)
    labels_batch = torch.tensor(labels_batch)
    return node_features_batch, adj_matrix_batch, labels_batch
