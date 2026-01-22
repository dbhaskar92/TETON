import pandas as pd
import torch
from model import TemporalSCCN_approach1, TemporalSCCN_approach2
import argparse
import numpy as np
import os
from data_processing import EEGDatasetCached, RandomEEGDatasetCached
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

warnings.filterwarnings(
    "ignore"
)
torch.manual_seed(0)

def sparse_collate(batch):
    return batch  # list of samples, no stacking

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total

def train_GD(model, device, train_loader, optimizer, loss_fn, epoch_writer, epoch, args):
    model.train()
    all_labels = torch.tensor([]).to(device)
    all_outputs = torch.tensor([]).to(device)
    model.zero_grad()
    for batch in tqdm(train_loader):
        windows = batch[0][0]
        label = batch[0][-1].to(device)
        if windows[0][0][0].shape[1] != args.win_len:
            continue
        else:
            output = model(windows)
            if label.dim() < 2:
                label = label.unsqueeze(0)
            all_labels = torch.cat((all_labels, label), dim=0)
            all_outputs = torch.cat((all_outputs, output), dim=0)
        torch.cuda.empty_cache()
    train_loss = loss_fn(all_outputs, all_labels.long())
    train_loss.backward()
    optimizer.step()
    epoch_writer.add_scalar('Train/Loss', train_loss.item(), epoch)
    correct_predictions = (all_outputs.argmax(dim=1) == all_labels.long()).sum().item()
    print(f"\n---------Epoch {epoch+1}/{args.epoch}---------")
    print(f"\nTrain Correct Predictions: {correct_predictions} out of {all_labels.size(0)}")
    #class distribution in all_labels
    unique, counts = torch.unique(all_labels, return_counts=True)
    print(f"Class distribution in train set: {dict(zip(unique.cpu().numpy().tolist(), counts.cpu().numpy().tolist()))}\n")   
    train_accuracy = correct_predictions / all_labels.size(0)
    epoch_writer.add_scalar('Train/Accuracy', train_accuracy, epoch)
    
def train_MB(model, device, train_loader, optimizer, loss_fn, epoch_writer, epoch, args):
    model.train()
    labels = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)
    all_outputs = torch.tensor([]).to(device)
    iteration = 0
    correct_predictions = 0
    num_iter = 0
    loader_writer = SummaryWriter(log_dir=f"save/{args.output_dir}/runs/epoch_{epoch}")
    model.zero_grad()
    for batch in tqdm(train_loader):
        windows = batch[0][0]
        label = batch[0][-1].to(device)
        if windows[0][0][0].shape[1] != args.win_len:
            continue
        else:
            output = model(windows)
            if label.dim() < 2:
                label = label.unsqueeze(0)
            all_labels = torch.cat((all_labels, label), dim=0)
            labels = torch.cat((labels, label), dim=0)
            all_outputs = torch.cat((all_outputs, output), dim=0)
            iteration += 1
        
        if iteration == args.batch_size:
            train_loss = loss_fn(all_outputs, all_labels.long())
            train_loss.backward()
            optimizer.step()
            correct_predictions+= (all_outputs.argmax(dim=1) == all_labels.long()).sum().item()
            loader_writer.add_scalar('Batch/Loss', train_loss.item(), num_iter)
            iteration = 0
            num_iter += 1
            all_labels = torch.tensor([]).to(device)
            all_outputs = torch.tensor([]).to(device)
            model.zero_grad()
        torch.cuda.empty_cache()
    loader_writer.close()
    print(f"\n---------Epoch {epoch+1}/{args.epoch}---------")
    print(f"\nTrain Correct Predictions: {correct_predictions} out of {labels.size(0)}")
    #class distribution in all_labels
    unique, counts = torch.unique(labels, return_counts=True)
    print(f"Class distribution in train set: {dict(zip(unique.cpu().numpy().tolist(), counts.cpu().numpy().tolist()))}\n")   
    train_accuracy = correct_predictions / labels.size(0)
    epoch_writer.add_scalar('Train/Accuracy', train_accuracy, epoch)
    
def train_SGD(model, device, train_loader, optimizer, loss_fn, epoch_writer, epoch, args):
    model.train()
    correct_predictions = 0
    iteration = 0
    total_loss = 0
    loader_writer = SummaryWriter(log_dir=f"save/{args.output_dir}/runs/epoch_{epoch}")
    for batch in train_loader:
        model.zero_grad()
        windows = batch[0][0]
        label = batch[0][-1].to(device)
        if windows[0][0][0].shape[1] != args.win_len:
            continue
        else:
            output = model(windows)
            if label.dim() < 2:
                label = label.unsqueeze(0)
            #print("Output shape:", output.shape, "Label shape:", label.shape)
            train_loss = loss_fn(output, label)
            train_loss.backward()
            optimizer.step()
            iteration += 1
            correct_predictions += (output.argmax(dim=1) == label).sum().item()
            total_loss += train_loss.item()
            loader_writer.add_scalar('Batch/Loss', train_loss.item(), iteration)

    loader_writer.close()
    print(f"\n-----------------Epoch {epoch+1}/{args.epoch}-----------------")
    epoch_writer.add_scalar('Train/Loss', total_loss / iteration, epoch)
    print(f"\n\nTrain Correct Predictions: {correct_predictions} out of {iteration}")  
    train_accuracy = correct_predictions / iteration
    epoch_writer.add_scalar('Train/Accuracy', train_accuracy, epoch)
    
def test(model, device, test_loader, loss_fn, epoch_writer, epoch, args):
    model.eval()
    all_labels = torch.tensor([]).to(device)
    all_outputs = torch.tensor([]).to(device)
    for batch in test_loader:
        windows = batch[0][0]
        if windows[0][0][0].shape[1] != args.win_len:
            print("Skipping test sample due to window length mismatch.")
            continue
        else:
            label = batch[0][-1].to(device)
            output = model(windows)
            if label.dim() < 2:
                label = label.unsqueeze(0)
            all_labels = torch.cat((all_labels, label), dim=0)
            all_outputs = torch.cat((all_outputs, output), dim=0)
    test_loss = loss_fn(all_outputs, all_labels.long())
    epoch_writer.add_scalar('Test/Loss', test_loss.item(), epoch)
    correct_predictions = (all_outputs.argmax(dim=1) == all_labels.long()).sum().item()
    print(f"Test Correct Predictions: {correct_predictions} out of {all_labels.size(0)}")
    #class distribution in all_labels
    unique, counts = torch.unique(all_labels, return_counts=True)
    print(f"Class distribution in test set: {dict(zip(unique.cpu().numpy().tolist(), counts.cpu().numpy().tolist()))}\n")
    test_accuracy = correct_predictions / all_labels.size(0)
    epoch_writer.add_scalar('Test/Accuracy', test_accuracy, epoch)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--csv_file', type=str, required=True, help='Path to input CSV file')
    #parser.add_argument('--num_nodes', type=int, required=True, help='Number of nodes in the system')
    parser.add_argument('--input_dim', type=int, default=3, help='Input dimension per node (default: 3 for Lorenz)')
    parser.add_argument('--d_max', type=int, default=2, help='Maximum polynomial degree in library')
    parser.add_argument('--win_len', type=int, default=1024, help='Window length for local SINDy')
    parser.add_argument('--max_rows', type=int, default=6000, help='Maximum number of rows to use in local SINDy')
    parser.add_argument('--rho_val', type=float, default=0.5, help='Rho value for hierarchical SOC constraint')
    parser.add_argument('--scale', type=float, default=1.2, help='Scaling factor for lambda')
    parser.add_argument('--admm_rho', type=float, default=3.0, help='Initial ADMM rho value')
    parser.add_argument('--admm_overrelax', type=float, default=1.6, help='ADMM over-relaxation parameter')
    parser.add_argument('--max_iters', type=int, default=50, help='Maximum ADMM iterations')
    parser.add_argument('--row_norm_nnz_thr', type=float, default=1e-6, help='Row norm threshold for non-zero coefficients')
    parser.add_argument('--param_abs_nnz_thr', type=float, default=1e-6, help='Parameter absolute value threshold for non-zero coefficients')
    parser.add_argument('--q', type=float, default=0.5, help='Quantile for thresholding edges and triangles')
    parser.add_argument('--S3_mode', type=str, choices=['max', 'mean', 'geom'], default='max', help='Mode for triangle scoring')
    parser.add_argument('--win_sg', type=int, default=29, help='Savitzky-Golay window size for smoothing')
    parser.add_argument('--order', type=int, default=3, help='Savitzky-Golay polynomial order for smoothing')
    parser.add_argument('--sigma', type=float, default=10.0, help='Lorenz system sigma parameter')
    parser.add_argument('--pair_lambda_mult', type=float, default=1.0, help='Multiplier for pairwise terms in degree weights')
    parser.add_argument('--r_target_pc', type=float, default=0.95)
    parser.add_argument('--k_max', type=int, default=1000000)
    parser.add_argument('--k_min', type=int, default=600)
    parser.add_argument('--FS', type=float, default=256)
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--feature_aggr', type=str, choices=['mean', 'sum', 'max'], default='mean', help='Feature aggregation method')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default='test_1')
    parser.add_argument('--train_method', type=str, choices=['GD', 'SGD', 'MB'], default='SGD')
    parser.add_argument('--model_type', type=str, choices=['approach1', 'approach2'], default='approach1')
    parser.add_argument('--epoch', type=int, default=10)
    args = parser.parse_args()
    
    args.dt = 1/args.FS
    args.stride = 512
    args.half = (args.win_sg - 1) // 2

    #check if args.output_dir exists, if not create it
    if not os.path.exists(f"save/{args.output_dir}"):
        os.makedirs(f"save/{args.output_dir}")
    dataset = EEGDatasetCached(args)
    #dataset = RandomEEGDatasetCached(
    #cache_dir="random_processed",
    #num_samples=10,
    #num_timesteps=130,
    #num_channels=30,
    #num_classes=2,
    #args=args)
    #split dataset into train and test
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=sparse_collate)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=sparse_collate)
    if args.model_type == 'approach1':
        model = TemporalSCCN_approach1(in_channel=args.win_len, hidden_channels=768, out_channels=2).to(device)
    else:
        model = TemporalSCCN_approach2(in_channel=args.win_len, hidden_channels=768, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    epoch_writer = SummaryWriter(log_dir=f"save/{args.output_dir}/runs/overall")
    for epoch in range(args.epoch):
        if args.train_method == 'GD':
            train_GD(model, device, train_loader, optimizer, loss_fn, epoch_writer, epoch, args)
        elif args.train_method == 'MB':
            train_MB(model, device, train_loader, optimizer, loss_fn, epoch_writer, epoch, args)
        else:
            train_SGD(model, device, train_loader, optimizer, loss_fn, epoch_writer, epoch, args)
        test(model, device, test_loader, loss_fn, epoch_writer, epoch, args)
        print(f"----------------------------------------------\n")
        