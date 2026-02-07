#!/usr/bin/env python3
"""
Event-based experiments on SINDy topology for SCCN input preparation and training.

This script:
1) Supports three tasks:
   - TD vs FEP
   - Direct Gaze vs Diverted Gaze
   - Positive Movie vs Negative Movie
2) Builds SCCN-ready topological snapshots (features, incidences, adjacencies)
   from per-window SINDy edges/triangles using utils.construct_topological_snapshot.
3) Trains a Temporal SCCN model for classification.

Expected inputs from processed_eeg_dataset/:
- X_samples.pkl
- y_labels.npy
- metadata.csv
- sindy_topology.pkl
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings

warnings.filterwarnings(
    "ignore"
)

from utils import construct_topological_snapshot
from model import TemporalSCCN_approach1, TemporalSCCN_approach2

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def load_processed_data(processed_dir):
    x_path = os.path.join(processed_dir, "X_samples.pkl")
    y_path = os.path.join(processed_dir, "y_labels.npy")
    meta_path = os.path.join(processed_dir, "metadata.csv")
    sindy_path = os.path.join(processed_dir, "sindy_topology.pkl")

    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Missing {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Missing {y_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}")
    if not os.path.exists(sindy_path):
        raise FileNotFoundError(f"Missing {sindy_path}")

    with open(x_path, "rb") as f:
        X = pickle.load(f)
    y = np.load(y_path)
    metadata_df = pd.read_csv(meta_path)
    with open(sindy_path, "rb") as f:
        sindy_data = pickle.load(f)

    return X, y, metadata_df, sindy_data


def get_task_from_user():
    print("Select experiment task:")
    print("  1) TD vs FEP")
    print("  2) Direct Gaze vs Diverted Gaze")
    print("  3) Positive Movie vs Negative Movie")
    choice = input("Enter 1, 2, or 3: ").strip()
    mapping = {"1": "td_vs_fep", "2": "gaze", "3": "valence"}
    return mapping.get(choice, "td_vs_fep")


def label_from_metadata(task, group, event_type):
    """
    Conditions:
      1 = Direct Gaze + Positive Movie
      2 = Direct Gaze + Negative Movie
      3 = Diverted Gaze + Positive Movie
      4 = Diverted Gaze + Positive Movie (as provided)
    """
    if task == "td_vs_fep":
        return 0 if group == "FEP" else 1

    # Direct vs Diverted
    if task == "gaze":
        return 0 if event_type in (1, 2) else 1

    # Positive vs Negative
    # Condition 1, 3 = Positive; Condition 2, 4 = Negative
    if task == "valence":
        return 0 if event_type in (1, 3) else 1

    return 0


def build_snapshots(X, metadata_df, sindy_data, task, min_edges=1, min_triangles=1, max_windows=None):
    """
    Build topological snapshots for each window, grouped by sample (event).
    Classification is at the SAMPLE level, not window level.
    """
    sindy_edges = sindy_data.get("sindy_edges_per_sample", [])
    sindy_triangles = sindy_data.get("sindy_triangles_per_sample", [])

    if len(sindy_edges) != len(X) or len(sindy_triangles) != len(X):
        raise ValueError("SINDy results do not align with X_samples length.")

    snapshots = []
    sample_labels = []  # One label per sample (event), not per window
    windows_kept = 0
    windows_skipped = 0
    samples_kept = 0
    samples_skipped = 0

    for sample_idx in tqdm(range(len(X)), desc="Building snapshots"):
        sample_windows = X[sample_idx]
        sample_edges = sindy_edges[sample_idx]
        sample_triangles = sindy_triangles[sample_idx]

        num_windows = min(len(sample_windows), len(sample_edges), len(sample_triangles))
        if max_windows is not None:
            num_windows = min(num_windows, max_windows)

        group = metadata_df.loc[sample_idx, "group"]
        event_type = int(metadata_df.loc[sample_idx, "event_type"])
        label = label_from_metadata(task, group, event_type)

        sample_snapshots = []
        for w_idx in range(num_windows):
            edges = sample_edges[w_idx]
            triangles = sample_triangles[w_idx]

            if len(edges) < min_edges or len(triangles) < min_triangles:
                windows_skipped += 1
                continue

            # Node features: channels x timepoints
            # Window shape: (window_length, num_channels)
            window = sample_windows[w_idx]
            node_features = torch.tensor(window.T, dtype=torch.float32)

            try:
                features, incidences, adjacencies = construct_topological_snapshot(
                    node_features=node_features,
                    edge_list=edges,
                    triangle_list=triangles,
                    agg_func="mean",
                )
            except Exception:
                windows_skipped += 1
                continue

            sample_snapshots.append({
                "sample_idx": sample_idx,
                "window_idx": w_idx,
                "features": features,
                "incidences": incidences,
                "adjacencies": adjacencies,
                "group": group,
                "event_type": event_type,
            })
            windows_kept += 1

        # Only add sample if it has at least one valid window
        if len(sample_snapshots) > 0:
            snapshots.append({
                "sample_idx": sample_idx,
                "windows": sample_snapshots,
                "group": group,
                "event_type": event_type,
                "label": label,
            })
            sample_labels.append(label)
            samples_kept += 1
        else:
            samples_skipped += 1

    stats = {
        "samples_kept": samples_kept,
        "samples_skipped": samples_skipped,
        "windows_kept": windows_kept,
        "windows_skipped": windows_skipped,
    }
    return snapshots, np.array(sample_labels), stats


def summarize(labels, task):
    """Summarize sample-level (event-level) label distribution."""
    unique, counts = np.unique(labels, return_counts=True)
    label_dist = dict(zip(unique.tolist(), counts.tolist()))

    if task == "td_vs_fep":
        name_map = {0: "FEP", 1: "TD"}
    elif task == "gaze":
        name_map = {0: "Direct", 1: "Diverted"}
    else:
        name_map = {0: "Positive", 1: "Negative"}

    readable = {name_map.get(k, k): v for k, v in label_dist.items()}
    return readable


# ============================================================================
# Dataset Class for SCCN Training
# ============================================================================

class SnapshotDataset(Dataset):
    """Dataset for sample-level (event-level) classification with temporal SCCN."""
    
    def __init__(self, snapshots, labels):
        """
        Args:
            snapshots: list of sample dicts, each containing 'windows' list
            labels: numpy array of sample-level labels (one per sample/event)
        """
        self.snapshots = snapshots
        self.labels = labels
    
    def __len__(self):
        return len(self.snapshots)
    
    def __getitem__(self, idx):
        sample = self.snapshots[idx]
        windows = sample['windows']
        label = self.labels[idx]
        
        # Sort windows by window_idx
        windows = sorted(windows, key=lambda x: x['window_idx'])
        
        # Build temporal sequence of (features, incidences, adjacencies)
        temporal_data = []
        for win in windows:
            features = win['features']
            incidences = win['incidences']
            adjacencies = win['adjacencies']
            temporal_data.append((features, incidences, adjacencies))
        
        return temporal_data, torch.tensor(label, dtype=torch.long)


def sparse_collate(batch):
    """Collate function for sparse SCCN data."""
    return batch


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch_SGD(model, train_loader, optimizer, loss_fn, epoch, writer, win_len):
    """Train for one epoch using SGD (per-sample updates)."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
        temporal_data, label = batch[0]
        label = label.to(device)
        
        # Check window length
        if len(temporal_data) == 0:
            continue
        first_window_features = temporal_data[0][0]
        if first_window_features[0].shape[1] != win_len:
            continue
        
        optimizer.zero_grad()
        output = model(temporal_data)
        
        if label.dim() == 0:
            label = label.unsqueeze(0)
        
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += label.size(0)
        
        torch.cuda.empty_cache()
    
    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    
    writer.add_scalar('Train/Loss', avg_loss, epoch)
    writer.add_scalar('Train/Accuracy', accuracy, epoch)
    
    return avg_loss, accuracy


def train_epoch_MB(model, train_loader, optimizer, loss_fn, epoch, writer, win_len, batch_size, output_dir):
    """Train for one epoch using mini-batch gradient descent.
    
    Accumulates gradients over batch_size samples before updating weights.
    DataLoader batch_size remains 1, but effective batch_size is controlled here.
    """
    model.train()
    
    # Accumulators for mini-batch
    all_labels = torch.tensor([]).to(device)
    all_outputs = torch.tensor([]).to(device)
    epoch_labels = torch.tensor([]).to(device)  # Track all labels for epoch stats
    
    total_loss = 0
    iteration = 0
    correct_predictions = 0
    num_batches = 0
    
    batch_writer = SummaryWriter(log_dir=os.path.join(output_dir, "runs", f"epoch_{epoch}"))
    model.zero_grad()
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training (MB)"):
        temporal_data, label = batch[0]
        label = label.to(device)
        
        # Check window length
        if len(temporal_data) == 0:
            continue
        first_window_features = temporal_data[0][0]
        if first_window_features[0].shape[1] != win_len:
            continue
        
        output = model(temporal_data)
        
        if label.dim() == 0:
            label = label.unsqueeze(0)
        
        # Accumulate outputs and labels
        all_labels = torch.cat((all_labels, label), dim=0)
        epoch_labels = torch.cat((epoch_labels, label), dim=0)
        all_outputs = torch.cat((all_outputs, output), dim=0)
        iteration += 1
        
        # When we've accumulated batch_size samples, update weights
        if iteration == batch_size:
            train_loss = loss_fn(all_outputs, all_labels.long())
            total_loss += train_loss.item()
            train_loss.backward()
            optimizer.step()
            
            correct_predictions += (all_outputs.argmax(dim=1) == all_labels.long()).sum().item()
            batch_writer.add_scalar('Batch/Loss', train_loss.item(), num_batches)
            
            # Reset accumulators
            iteration = 0
            num_batches += 1
            all_labels = torch.tensor([]).to(device)
            all_outputs = torch.tensor([]).to(device)
            model.zero_grad()
        
        torch.cuda.empty_cache()
    '''
    # Handle remaining samples that didn't fill a complete batch
    if iteration > 0 and all_outputs.size(0) > 0:
        train_loss = loss_fn(all_outputs, all_labels.long())
        total_loss += train_loss.item()
        train_loss.backward()
        optimizer.step()
        correct_predictions += (all_outputs.argmax(dim=1) == all_labels.long()).sum().item()
        batch_writer.add_scalar('Batch/Loss', train_loss.item(), num_batches)'''
    
    batch_writer.close()
    
    total_samples = int(epoch_labels.size(0))
    accuracy = correct_predictions / max(total_samples, 1)
    
    # Log class distribution
    unique_label, counts_label = torch.unique(epoch_labels, return_counts=True)
    class_dist_label = dict(zip(unique_label.cpu().numpy().tolist(), counts_label.cpu().numpy().tolist()))
    unique_pred, counts_pred = torch.unique(all_outputs.argmax(dim=1), return_counts=True)
    class_dist_pred = dict(zip(unique_pred.cpu().numpy().tolist(), counts_pred.cpu().numpy().tolist()))
    
    print(f"  Train samples: {total_samples}, Correct: {correct_predictions}")
    print(f"  Class distribution: {class_dist_label}")
    print(f"  Predicted distribution: {class_dist_pred}")
    
    writer.add_scalar('Train/Accuracy', accuracy, epoch)
    
    return total_loss/iteration, accuracy  # Loss is logged per-batch, not per-epoch


def train_epoch(model, train_loader, optimizer, loss_fn, epoch, writer, win_len, 
                train_method='SGD', batch_size=4, output_dir=None):
    """Wrapper to select training method."""
    if train_method == 'MB':
        return train_epoch_MB(model, train_loader, optimizer, loss_fn, epoch, writer, 
                              win_len, batch_size, output_dir)
    else:
        return train_epoch_SGD(model, train_loader, optimizer, loss_fn, epoch, writer, win_len)


def evaluate(model, test_loader, loss_fn, epoch, writer, win_len):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Epoch {epoch+1} Evaluation"):
            temporal_data, label = batch[0]
            label = label.to(device)
            
            if len(temporal_data) == 0:
                continue
            first_window_features = temporal_data[0][0]
            if first_window_features[0].shape[1] != win_len:
                continue
            
            output = model(temporal_data)
            
            if label.dim() == 0:
                label = label.unsqueeze(0)
            
            loss = loss_fn(output, label)
            total_loss += loss.item()
            
            pred = output.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    
    writer.add_scalar('Test/Loss', avg_loss, epoch)
    writer.add_scalar('Test/Accuracy', accuracy, epoch)
    unique_label, counts_label = torch.unique(torch.tensor(all_labels), return_counts=True)
    class_dist_label = dict(zip(unique_label.cpu().numpy().tolist(), counts_label.cpu().numpy().tolist()))
    
    unique_pred, counts_pred = torch.unique(torch.tensor(all_preds), return_counts=True)
    class_dist_pred = dict(zip(unique_pred.cpu().numpy().tolist(), counts_pred.cpu().numpy().tolist()))
    
    print(f"  Test samples: {total}, Correct: {correct}")
    print(f"  Class distribution: {class_dist_label}")
    print(f"  Predicted distribution: {class_dist_pred}")
    
    return avg_loss, accuracy, all_preds, all_labels


def compute_class_weights(labels):
    """Compute sample weights for balanced sampling."""
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in labels]
    return sample_weights


def main():
    parser = argparse.ArgumentParser(description="Prepare SCCN-ready snapshots from SINDy outputs and train model.")
    parser.add_argument("--processed_dir", type=str, default="processed_eeg_dataset",
                        help="Directory with processed dataset and SINDy outputs")
    parser.add_argument("--task", type=str, choices=["td_vs_fep", "gaze", "valence"],
                        default=None, help="Experiment task")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for prepared snapshots and model")
    parser.add_argument("--min_edges", type=int, default=1, help="Minimum edges required per window")
    parser.add_argument("--min_triangles", type=int, default=1, help="Minimum triangles required per window")
    parser.add_argument("--max_windows", type=int, default=None, help="Max windows per sample")
    parser.add_argument("--fix_imbalanced", action="store_true", help="Use weighted sampling to fix class imbalance")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_channels", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of SCCN layers")
    parser.add_argument("--model_type", type=str, choices=["approach1", "approach2"], 
                        default="approach1", help="Model architecture")
    parser.add_argument("--train_split", type=float, default=0.7, help="Training split ratio")
    parser.add_argument("--win_len", type=int, default=768, help="Window length for model")
    parser.add_argument("--skip_training", action="store_true", help="Only build snapshots, skip training")
    parser.add_argument("--train_method", type=str, choices=["SGD", "MB"], default="SGD",
                        help="Training method: SGD (per-sample) or MB (mini-batch)")
    parser.add_argument("--batch_size", type=int, default=4, help="Effective batch size for MB training")
    
    args = parser.parse_args()

    task = args.task or get_task_from_user()
    output_dir = args.output_dir or os.path.join("save", f"experiment_{task}")
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # Step 1: Load and build snapshots
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"TASK: {task.upper().replace('_', ' ')}")
    print("=" * 70)
    
    X, y, metadata_df, sindy_data = load_processed_data(args.processed_dir)

    print(f"\nProcessed dir: {args.processed_dir}")
    print(f"Output dir: {output_dir}")

    snapshots, labels, stats = build_snapshots(
        X,
        metadata_df,
        sindy_data,
        task,
        min_edges=args.min_edges,
        min_triangles=args.min_triangles,
        max_windows=args.max_windows,
    )

    # Save snapshots
    snapshot_dir = os.path.join(output_dir, "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)
    with open(os.path.join(snapshot_dir, "snapshots.pkl"), "wb") as f:
        pickle.dump(snapshots, f)
    np.save(os.path.join(snapshot_dir, "labels.npy"), labels)

    print("\n" + "=" * 70)
    print("SNAPSHOT SUMMARY (Sample/Event Level)")
    print("=" * 70)
    print(f"Samples (events) kept: {stats['samples_kept']}")
    print(f"Samples (events) skipped: {stats['samples_skipped']}")
    print(f"Total windows kept: {stats['windows_kept']}")
    print(f"Total windows skipped: {stats['windows_skipped']}")
    print(f"Sample-level label distribution: {summarize(labels, task)}")
    
    if args.skip_training:
        print("\nSkipping training (--skip_training flag set)")
        return

    # =========================================================================
    # Step 2: Create Dataset and DataLoaders
    # =========================================================================
    print("\n" + "=" * 70)
    print("PREPARING DATA LOADERS")
    print("=" * 70)
    
    dataset = SnapshotDataset(snapshots, labels)
    
    # Split into train/test
    n_samples = len(dataset)
    n_train = int(args.train_split * n_samples)
    n_test = n_samples - n_train
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Total samples (events): {n_samples}")
    print(f"Train samples: {n_train}")
    print(f"Test samples: {n_test}")
    
    # Get labels for training set for weighted sampling
    train_indices = train_dataset.indices
    train_labels = np.array([labels[i] for i in train_indices])
    
    # Weighted sampler for class balance
    sample_weights = compute_class_weights(train_labels)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    if args.fix_imbalanced:    
        train_loader = DataLoader(train_dataset, batch_size=1, sampler=sampler, collate_fn=sparse_collate)
    else:
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=sparse_collate)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=sparse_collate)

    # =========================================================================
    # Step 3: Initialize Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("INITIALIZING MODEL")
    print("=" * 70)
    
    if args.model_type == "approach1":
        model = TemporalSCCN_approach1(
            in_channel=args.win_len,
            hidden_channels=args.hidden_channels,
            out_channels=2,
            n_layers=args.n_layers
        ).to(device)
    else:
        model = TemporalSCCN_approach2(
            in_channel=args.win_len,
            hidden_channels=args.hidden_channels,
            out_channels=2,
            n_layers=args.n_layers
        ).to(device)
    
    print(f"Model: {args.model_type}")
    print(f"Hidden channels: {args.hidden_channels}")
    print(f"SCCN layers: {args.n_layers}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training method: {args.train_method}")
    if args.train_method == 'MB':
        print(f"Effective batch size: {args.batch_size}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "runs"))

    # =========================================================================
    # Step 4: Training Loop
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    best_accuracy = 0.0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, loss_fn, epoch, writer, args.win_len,
            train_method=args.train_method, batch_size=args.batch_size, output_dir=output_dir
        )
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        
        test_loss, test_acc, preds, true_labels = evaluate(
            model, test_loader, loss_fn, epoch, writer, args.win_len
        )
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_epoch = epoch + 1
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy': test_acc,
                'task': task,
            }, os.path.join(output_dir, "best_model.pt"))
    
    writer.close()
    
    # =========================================================================
    # Step 5: Final Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Task: {task}")
    print(f"Best Test Accuracy: {best_accuracy:.4f} (Epoch {best_epoch})")
    print(f"Model saved to: {os.path.join(output_dir, 'best_model.pt')}")
    print(f"TensorBoard logs: {os.path.join(output_dir, 'runs')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
