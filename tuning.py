import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from datasets import SHL_Dataset, Alzh_Dataset, TE_Dataset, collate_fn_ft
from models import Brain_GCN, EEG_model
from prompt import GraphPrompt, GPFplusAtt, EdgeMask
from utils import set_seed
import os
from tqdm import tqdm
import numpy as np
import argparse

def main(args):
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if os.path.exists(args.dataset):
        dataset = torch.load(args.dataset)
    elif args.name == 'shl':
        dataset = SHL_Dataset('raw/SHL Datasets',sample_rate=200,window_length=10000)
    elif args.name == 'shl':
        dataset = Alzh_Dataset('raw/Alzh Datasets',sample_rate=200,window_length=10000)
    elif args.name == 'shl':
        dataset = TE_Dataset('raw/Turkish Epilepsy Dataset',sample_rate=200,window_length=10000)
    else:
        raise ValueError('Dataset not exists.')
    Fold = 5
    split = StratifiedShuffleSplit(n_splits=Fold, test_size=0.2, random_state=args.seed)

    input_dim = 5000  
    hidden_dim = 1024
    classes = args.classes
    num_layers = args.layer
    drop_ratio = 0

    avg_test_performance = [0, 0, 0, 0, 0]
    print('\n',args,'\n')

    full_labels = None
    full_scores = None
    full_predict = None

    for fold, (train_index, test_index) in enumerate(split.split(dataset.data, dataset.labels)):

        encoder = Brain_GCN(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, drop_ratio=drop_ratio, graph_pooling=args.ro, gtype=args.gtype).to(device)
        if not os.path.exists(args.model_path):
            args.tuning = 'fine-tune'
            args.fp = 'None'
            args.sp = 0
        else:
            encoder.load_state_dict(torch.load(args.model_path))
            print(f"Load model: {args.model_path}")
        model = EEG_model(encoder, input_dim, hidden_dim, classes, drop_out=0.5).to(device)

        best_test_performance = (float('inf'), 0, 0, 0, 0)  # (loss, accuracy, roc_auc, f1)

        best_l = None
        best_s = None
        best_p = None

        print(f"Fold {fold}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")

        full_train_subset = Subset(dataset, train_index)
        val_ratio = 0.25
        train_indices, val_indices = train_test_split(full_train_subset.indices, test_size=val_ratio, random_state=args.seed)

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        test_subset = Subset(dataset, test_index)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_ft)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_ft)
        test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_ft)

        model_param_group = []
        prompt = None
        edge_mask = None

        if args.tuning == 'gepl':
            print(f'Tuning method: {args.tuning}, fp: {args.fp}, sp: {args.sp}')
            print('params: MLP.parameters()')
            model_param_group.append({"params": model.MLP.parameters(), "lr": args.lr})
            
            for param in model.encoder.parameters():
                param.requires_grad = False
            
            if args.fp != 'None':
                if args.fp == 'gpf':
                    prompt = GraphPrompt(input_dim).to(device)    
                elif args.fp == 'gpf-plus':
                    prompt = GPFplusAtt(input_dim, p_num=args.p_num).to(device)
                else:
                    raise ValueError("prompt feature missing")
                print('params: prompt.parameters()')
                model_param_group.append({"params": prompt.parameters(), "lr": args.lr})
            
            if args.sp == 1:
                edge_mask = EdgeMask(dataset[0][0][0].shape[0]).to(device)
                print('params: edge_mask.parameters()')
                model_param_group.append({"params": edge_mask.parameters(), "lr": args.lr})
            else:
                edge_mask = None

        elif args.tuning == 'fine-tune':
            print(f'Tuning method: {args.tuning}')
            print('params: model.parameters()')
            model_param_group.append({"params": model.parameters(), "lr": args.lr})

        elif args.tuning == 'MLP':
            print(f'Tuning method: {args.tuning}')
            print('params: MLP.parameters()')
            model_param_group.append({"params": model.MLP.parameters(), "lr": args.lr})
            for param in model.encoder.parameters():
                param.requires_grad = False

        elif args.tuning == 'PARTIAL':
            print(f'Tuning method: {args.tuning}')
            print('params: MLP.parameters()')
            model_param_group.append({"params": model.MLP.parameters(), "lr": args.lr})

            print(f'params: Last {args.partial_k} layers of model.encoder.parameters()')

            for layer_id in range(args.partial_k):
                model_param_group.append({"params": model.encoder.convs[-layer_id].parameters(), "lr": args.lr})
        
        else:
            raise ValueError("--tuning: gepl/fine-tune/MLP-k/PARTIAL-k")

        optimizer = Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
        if args.lr_decay:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        else:
            scheduler = None

        for epoch in range(args.num_epochs):
            train_loss = train(model, train_loader, optimizer, device, epoch, prompt, edge_mask)
            val_loss, val_accuracy, val_roc_auc, val_f1 = evaluate(model, val_loader, device, prompt, edge_mask)
            test_loss, test_accuracy, test_roc_auc, test_f1, l, s, p = evaluate(model, test_loader, device, prompt, edge_mask, show=False, scores=True)

            print(f'Fold {fold+1}, Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}, Validation AUC: {val_roc_auc}, Validation F1: {val_f1}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Test AUC: {test_roc_auc}, Test F1: {test_f1}')

            if scheduler is not None:
                scheduler.step()

            # Update best performance
            if val_loss < best_test_performance[0] :
                best_test_performance = (val_loss, test_loss, test_accuracy, test_roc_auc, test_f1)
                best_l = l
                best_s = s
                best_p = p
        
        avg_test_performance[1] += best_test_performance[1]
        avg_test_performance[2] += best_test_performance[2]
        avg_test_performance[3] += best_test_performance[3]
        avg_test_performance[4] += best_test_performance[4]

        if full_labels is None:
            full_labels = best_l
            full_scores = best_s
            full_predict = best_p
        else:
            full_labels = np.concatenate([full_labels,best_l])
            full_scores = np.concatenate([full_scores,best_s])
            full_predict = np.concatenate([full_predict,best_p])

    print('ALL Labels:')
    print(full_labels)
    print('ALL Scores:')
    print(full_scores)
    print('ALL Predict:')
    print(full_predict)

    print("Avg Test Performance across all folds:")
    print(f"Test Loss: {avg_test_performance[1]/Fold}, Test Accuracy: {avg_test_performance[2]/Fold}, Test AUC: {avg_test_performance[3]/Fold}, Test F1: {avg_test_performance[4]/Fold}")

def train(model, train_loader, optimizer, device, epoch, prompt, edge_mask):
    model.train()
    if prompt is not None: prompt.train()
    if edge_mask is not None: edge_mask.train()

    total_loss = 0
    for batch in tqdm(train_loader, desc=f'Training Epoch {epoch+1}'):
        inputs, adj, labels = batch
        inputs, adj, labels = inputs.to(device), adj.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, adj, fp=prompt, sp=edge_mask)
        loss = F.cross_entropy(outputs, labels.long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss

def evaluate(model, val_loader, device, prompt, edge_mask, show=False, scores=False):
    model.eval()
    if prompt is not None: prompt.eval()
    if edge_mask is not None: edge_mask.eval()

    val_loss = 0
    correct = 0
    total = 0

    all_predictions = []
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            inputs, adj, labels = batch
            inputs, adj, labels = inputs.to(device), adj.to(device), labels.to(device)

            outputs = model(inputs, adj, fp=prompt, sp=edge_mask)
            prob = F.softmax(outputs, dim=1)

            val_loss += F.cross_entropy(outputs, labels.long(), reduction='sum').item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().tolist())
            all_scores.append(prob[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().tolist())

    all_labels = [int(i) for i in all_labels]
    all_scores = np.concatenate(all_scores)
    all_predictions = [int(i) for i in all_predictions]
    if show:
        print(f"Actual   : {all_labels}")
        print(f"Predicted: {all_predictions}")

    val_loss /= len(val_loader.dataset)
    accuracy = correct / total
    roc_auc = roc_auc_score(all_labels, all_scores, multi_class='ovr')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    if scores == False:
        return val_loss, accuracy, roc_auc, f1
    else:
        return val_loss, accuracy, roc_auc, f1, np.array(all_labels), all_scores, np.array(all_predictions)

def get_args():
    parser = argparse.ArgumentParser(description="Training and Evaluation Script")
    parser.add_argument('--name', type=str, default='shl', help='Dataset name')
    parser.add_argument('--classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--fp', type=str, default='gpf', help='Prompt type')
    parser.add_argument('--p_num', type=int, default=4, help='Prompt number')
    parser.add_argument('--sp', type=int, default=1, help='Use edge mask')

    parser.add_argument('--tuning', type=str, default='gepl', help='tuning method, gepl/fine-tune/MLP/PARTIAL')
    
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=500, help='Early stopping patience')
    parser.add_argument('--ro', type=str, default='mean', help='readout function')
    parser.add_argument('--model_path', type=str, default='saved_models/encoder.pth', help='Pretrain model')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument('--dataset', type=str, default='none', help='dataset')
    parser.add_argument('--partial_k', type=int, default=1, help='num of layers to fine tune in PARTIAL-k')
    parser.add_argument('--lr_decay', type=bool, default=True, help='learning rate decay')
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gtype", type=str, default='gcn')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)