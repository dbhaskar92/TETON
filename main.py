import pandas as pd
import torch
from model import TemporalSCCN
import argparse
import numpy as np
from data_processing import EEGDatasetCached, RandomEEGDatasetCached

from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

warnings.filterwarnings(
    "ignore"
)

def sparse_collate(batch):
    return batch  # list of samples, no stacking


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
    parser.add_argument('--win_len', type=int, default=50, help='Window length for local SINDy')
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
    parser.add_argument('--epoch', type=int, default=10)
    args = parser.parse_args()
    
    args.dt = 1/args.FS
    args.stride = args.win_len
    args.half = (args.win_sg - 1) // 2
    args.num_nodes = 31

    #dataset = EEGDatasetCached(args)
    dataset = RandomEEGDatasetCached(
    cache_dir="random_processed",
    num_samples=10,
    num_timesteps=130,
    num_channels=30,
    num_classes=2,
    args=args
)
    #split dataset into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=sparse_collate)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=sparse_collate)
    
    model = TemporalSCCN(in_channel=args.win_len, hidden_channels=512, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    progress_bar = tqdm(range(args.epoch), desc="Epoch", leave=False)
    for epoch in progress_bar:
        for batch in train_loader:
            model.zero_grad()
            windows = batch[0][0]
            label = batch[0][-1].to(device)
            output = model(windows)
            train_loss = loss_fn(label.float(), output.squeeze(0))
            train_loss.backward()
            optimizer.step()
            
        for batch in test_loader:
            windows = batch[0][0]
            label = batch[0][-1].to(device)
            output = model(windows)
            test_loss = loss_fn(label.float(), output.squeeze(0))
        
        progress_bar.set_postfix({'Train Loss': train_loss.item(), 'Test Loss': test_loss.item()})
        