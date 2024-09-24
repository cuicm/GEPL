import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
import argparse
from datasets import Pretrain_Dataset, collate_fn
from models import Brain_GCN, Encoder, Projection
from utils import set_seed, ChannelDropout, ConnectionDropout, FeatureMask

def train_model(augmentor, data_path, save_path, lr=0.0001, batch_size=32, num_epochs=100, seed=42):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = torch.load(data_path)
    print(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    input_dim = 5000  
    hidden_dim = 1024
    output_dim = 1024
    num_layers = 5
    drop_ratio = 0

    encoder_model = Brain_GCN(input_dim=input_dim,
                              hidden_dim=hidden_dim,
                              num_layers=num_layers,
                              drop_ratio=drop_ratio,
                              graph_pooling="add").to(device)

    encoder = Encoder(encoder_model, augmentor).to(device)
    projection = Projection(hidden_dim, output_dim).to(device)

    optimizer = Adam(list(encoder.parameters()) + list(projection.parameters()), lr=lr)

    def contrastive_loss(x1, x2, T=0.1):
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim) + 1e-4)
        loss = - torch.log(loss).mean() + 10
        return loss

    best_loss = float('inf')

    for epoch in range(num_epochs):
        encoder.train()
        projection.train()

        total_loss = 0
        step = 0
        for node_features_batch, adj_matrix_batch in dataloader:
            optimizer.zero_grad()

            batch_g1, batch_g2 = [], []
            for i in range(len(node_features_batch)):
                node_features_batch[i] = node_features_batch[i].to(device)
                adj_matrix_batch[i] = adj_matrix_batch[i].to(device)

                _, g, _, _, g1, g2 = encoder(node_features_batch[i], adj_matrix_batch[i])

                batch_g1.append(projection(g1))
                batch_g2.append(projection(g2))

            batch_g1 = torch.cat(batch_g1, dim=0)
            batch_g2 = torch.cat(batch_g2, dim=0)

            loss = contrastive_loss(batch_g1, batch_g2)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step += 1

        average_loss = total_loss / step
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss}")

        if average_loss < best_loss:
            best_loss = average_loss
            torch.save(encoder.encoder.state_dict(), f'{save_path}/encoder.pth')
            print(f"New best model saved with loss {best_loss}")

    print("Pre-training completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GCN model with contrastive learning")
    parser.add_argument("--augmentor", type=str, choices=["ConnectionDropout", "ChannelDropout", "FeatureMask"], default="ConnectionDropout", help="Type of augmentor to use")
    parser.add_argument("--data_path", type=str, default='PROCESSED/Pretrain.pth', help="Path to the dataset")
    parser.add_argument("--save_path", type=str, default='saved_models', help="Path to save the trained model")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    augmentor_mapping = {
        "ConnectionDropout": (ConnectionDropout(0.3), ConnectionDropout(0.3)),
        "ChannelDropout": (ChannelDropout(0.3), ChannelDropout(0.3)),
        "FeatureMask": (FeatureMask(0.3), FeatureMask(0.3))
    }

    augmentor = augmentor_mapping[args.augmentor]

    train_model(augmentor, args.data_path, args.save_path, args.lr, args.batch_size, args.num_epochs, args.seed)