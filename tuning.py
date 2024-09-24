import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, f1_score
from datasets import SHL_Dataset, Alzh_Dataset, TE_Dataset, collate_fn_ft
from models import Brain_GCN, EEG_model
from prompt import GraphPrompt, GPFplusAtt, EdgeMask
from utils import set_seed
import os
from tqdm import tqdm
import numpy as np

def main(args):
    set_seed(args.seed)
    print(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = torch.load(args.dataset)

    dataset_size = len(dataset)
    val_size = int(0.2 * dataset_size)
    test_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size - test_size

    train_subset, val_subset, test_subset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_ft)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_ft)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_ft)

    input_dim = 5000  
    hidden_dim = 1024
    classes = args.classes
    num_layers = 5
    drop_ratio = 0

    encoder = Brain_GCN(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, drop_ratio=drop_ratio, graph_pooling=args.ro).to(device)
    if not os.path.exists(args.model_path):
        print('No pretrained model')
        args.tuning = 'fine-tune'
        args.fp = 'None'
        args.sp = 0
    else:
        encoder.load_state_dict(torch.load(args.model_path))
        print(f"Load model: {args.model_path}")
    model = EEG_model(encoder, input_dim, hidden_dim, classes, drop_out=0.5).to(device)


    model_param_group = []
    prompt = None
    edge_mask = None


    if args.tuning == 'eeg-pro':
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
        raise ValueError("--tuning: eeg-pro/fine-tune/MLP-k/PARTIAL-k")

    print('\n',args,'\n')

    optimizer = Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    print(optimizer)
    total_params = 0
    for i in range(len(optimizer.param_groups)):
        total_params += sum(p.numel() for p in optimizer.param_groups[i]['params'])
    print(f"Total number of parameters in the optimizer: {total_params}")

    if args.lr_decay:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    else:
        scheduler = None

    best_val_loss = float('inf')
    best_test_performance = None
    best_val_performance = None
    patience = args.patience
    patience_counter = 0

    for epoch in range(args.num_epochs):
        train_loss = train(model, train_loader, optimizer, device, epoch, prompt, edge_mask)
        val_loss, val_accuracy, val_roc_auc, val_f1 = evaluate(model, val_loader, device, prompt, edge_mask)
        test_loss, test_accuracy, test_roc_auc, test_f1 = evaluate(model, test_loader, device, prompt, edge_mask, show=True)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}, Validation AUC: {val_roc_auc}, Validation F1: {val_f1}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Test AUC: {test_roc_auc}, Test F1: {test_f1}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_test_performance = (test_loss, test_accuracy, test_roc_auc, test_f1)
            best_val_performance = (val_loss, val_accuracy, val_roc_auc, val_f1)
            patience_counter = 0
        else:
            patience_counter += 1

        if scheduler is not None:
            scheduler.step()
        
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    if edge_mask != None:
        edge_mask.print_adj()

    print("Best Validation Performance and Corresponding Test Performance:")
    print(f"Validation Loss: {best_val_loss}")
    print(f"Val Accuracy: {best_val_performance[1]}, Val AUC: {best_val_performance[2]}, Val F1: {best_val_performance[3]}")
    
    print(f"Test Accuracy: {best_test_performance[1]}, Test AUC: {best_test_performance[2]}, Test F1: {best_test_performance[3]}")

def train(model, train_loader, optimizer, device, epoch, prompt, edge_mask):
    model.train()
    if prompt != None: prompt.train()
    if edge_mask != None: edge_mask.train()

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

def evaluate(model, val_loader, device, prompt, edge_mask, show=False):
    model.eval()
    if prompt != None: prompt.eval()
    if edge_mask != None: edge_mask.eval()

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

    all_labels=[int(i) for i in all_labels]
    all_scores=np.concatenate(all_scores)
    all_predictions=[int(i) for i in all_predictions]
    if show:
        print(f"Actual   : {all_labels}")
        print(f"Predicted: {all_predictions}")

    val_loss /= len(val_loader.dataset)
    accuracy = correct / total
    roc_auc = roc_auc_score(all_labels,all_scores)
    f1 = f1_score(all_labels, all_predictions)
    return val_loss, accuracy, roc_auc,f1

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Training and Evaluation Script")
    parser.add_argument('--name', type=str, default='shl', help='Dataset name')
    parser.add_argument('--classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--fp', type=str, default='gpf', help='Prompt type')
    parser.add_argument('--p_num', type=int, default=4, help='Prompt number')
    parser.add_argument('--sp', type=int, default=1, help='Use edge mask')

    parser.add_argument('--tuning', type=str, default='eeg-pro', help='tuning method, eeg-pro/fine-tune/MLP/PARTIAL')
    
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=500, help='Early stopping patience')
    parser.add_argument('--ro', type=str, default='add', help='readout function')
    parser.add_argument('--model_path', type=str, default='saved_models/encoder51.pth', help='Pretrain model')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument('--dataset', type=str, default='none', help='dataset')
    parser.add_argument('--partial_k', type=int, default=1, help='num of layers to fine tune in PARTIAL-k')
    parser.add_argument('--lr_decay', type=bool, default=False, help='learning rate decay')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)
