#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np
from glob import glob
from collections import defaultdict
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from scipy import sparse as sp
from topomodelx.nn.simplicial.sccnn import SCCNN
from topomodelx.utils.sparse import from_sparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def load_npz(path):
    """Load window data from NPZ file."""
    z = np.load(path, allow_pickle=True)
    
    def mk_sparse(rows, cols, data, shape):
        if shape[0]==0 or shape[1]==0 or data.size==0:
            return sp.coo_matrix(shape)
        return sp.coo_matrix((data, (rows, cols)), shape=tuple(shape))
    
    B1 = mk_sparse(z['B1_rows'], z['B1_cols'], z['B1_data'], z['B1_shape'])
    B2 = mk_sparse(z['B2_rows'], z['B2_cols'], z['B2_data'], z['B2_shape'])
    H0 = z['H0'].astype(np.float32)
    H1 = z['H1'].astype(np.float32)
    H2 = z['H2'].astype(np.float32)
    
    return {
        'B1': B1, 'B2': B2, 'H0': H0, 'H1': H1, 'H2': H2,
        'subject_id': str(z['subject_id']),
        'label': int(z['label']),
        'window_idx': int(z['window_idx']),
        't_mid': float(z['t_mid'])
    }


class SubjectDataset(Dataset):
    """Dataset that loads all windows for each subject for person-wise classification."""
    
    def __init__(self, data_dir, train_subjects=None, test_subjects=None, mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        
        # Load manifest
        with open(os.path.join(data_dir, 'manifest.json')) as f:
            self.manifest = json.load(f)
        
        # Filter subjects based on mode
        if mode == 'train':
            self.subjects = train_subjects or []
        else:
            self.subjects = test_subjects or []
        
        # Get window files for each subject from the manifest
        self.subject_windows = defaultdict(list)
        for window_info in self.manifest['windows']:
            subject_id = window_info['subject_id']
            if subject_id in self.subjects:  # Only include windows for subjects in this split
                self.subject_windows[subject_id].append(window_info)
        
        total_windows = sum(len(w) for w in self.subject_windows.values())
        print(f"{mode.capitalize()} dataset: {len(self.subjects)} subjects, {total_windows} total windows")
    
    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, idx):
        subject_id = self.subjects[idx]
        windows = self.subject_windows[subject_id]
        
        # Load all windows for this subject
        subject_data = []
        for window_info in windows:
            filepath = os.path.join(self.data_dir, window_info['file'])
            if os.path.exists(filepath):
                data = load_npz(filepath)
                subject_data.append(data)
            else:
                print(f"Warning: Window file not found: {filepath}")
        
        return {
            'subject_id': subject_id,
            'label': self.manifest['subjects'][subject_id]['label'],
            'windows': subject_data
        }


def collate_fn(batch):
    """Custom collate function to handle person-wise data."""
    # Each batch item is a person with multiple windows
    return batch


# ------------------ Model ------------------

class TemporalSCCNNClassifier(nn.Module):
    """Temporal Simplicial Complex CNN Classifier for Brain Connectivity Analysis."""
    
    def __init__(self, d0, d1, d2, hidden_edge=64, gru_hidden=128, num_classes=2, dropout=0.3):
        super().__init__()
        
        # SCCNN: Simplicial Complex Convolutional Neural Network
        # Perfect for our multi-scale simplicial data (nodes, edges, triangles)
        # Note: SCCNN expects all input channels to match hidden channels
        self.sccnn = SCCNN(
            in_channels_all=[1, 1, 1],  # [1, 1, 1] - all features have 1 channel
            hidden_channels_all=[hidden_edge, hidden_edge, hidden_edge],  # [64, 64, 64] for each simplex level
            conv_order=2,  # Convolution order
            sc_order=2,    # Simplicial complex order
            n_layers=2     # Number of layers
        )
        
        # Temporal processing
        self.gru = nn.GRU(input_size=hidden_edge, hidden_size=gru_hidden, batch_first=True, dropout=dropout)
        
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, person_data):
        """
        Forward pass for person-wise classification using SCCNN.
        person_data: dict with 'windows' key containing list of window data
        """
        device = next(self.parameters()).device
        per_window_vec = []
        
        # Process each window for this person
        for window_data in person_data['windows']:
            # Convert to torch tensors
            B1 = from_sparse(window_data['B1']).to(device)
            B2 = from_sparse(window_data['B2']).to(device)
            H0 = torch.tensor(window_data['H0'], dtype=torch.float32).to(device)
            H1 = torch.tensor(window_data['H1'], dtype=torch.float32).to(device)
            H2 = torch.tensor(window_data['H2'], dtype=torch.float32).to(device)
            
            # Ensure B1 and B2 are 2D for matrix multiplication
            if B1.dim() > 2:
                B1 = B1.squeeze(0)  # Remove batch dimension if present
            if B2.dim() > 2:
                B2 = B2.squeeze(0)  # Remove batch dimension if present
            
            # Additional safety check - ensure they are exactly 2D
            if B1.dim() != 2:
                B1 = B1.view(B1.shape[-2], B1.shape[-1])
            if B2.dim() != 2:
                B2 = B2.view(B2.shape[-2], B2.shape[-1])
            
            # Prepare input for SCCNN
            # SCCNN expects: x_all = (node_features, edge_features, triangle_features)
            # Each feature needs channel dimension and must be on the same device
            x_all = (
                H0.unsqueeze(1).to(device),    # [30, 1] - node features with channel dim
                H1.to(device),                  # [363, 1] - edge features (already has channel dim)
                H2.to(device)                   # [64, 1] - triangle features (already has channel dim)
            )
            
            # Ensure B1 and B2 are dense tensors for matrix multiplication
            if B1.is_sparse:
                B1 = B1.to_dense()
            if B2.is_sparse:
                B2 = B2.to_dense()
            
            # Validate dimensions to avoid edge cases
            n_nodes, n_edges = B1.shape
            n_edges_check, n_triangles = B2.shape
            
            # Instead of rejecting windows with few triangles, accept them
            if n_triangles == 0:  # Only reject completely empty windows
                continue
            
            # Additional validation: ensure we have enough structure for SCCNN
            if n_edges < 10:  # Need at least 10 edges for meaningful structure
                print(f"Warning: Window {window_data.get('window_idx', 'unknown')} has only {n_edges} edges, skipping")
                continue
            
            if n_triangles < 2:  # Need at least 2 triangles for meaningful structure
                print(f"Warning: Window {window_data.get('window_idx', 'unknown')} has only {n_triangles} triangles, skipping")
                continue
            
            # Compute laplacians with correct dimensions
            # L0: Graph laplacian (nodes × nodes)
            L0 = torch.mm(B1, B1.t())      # [30 × 30] - nodes × nodes
            
            # L1_d: Down edge laplacian (edges × edges) - this should be [379 × 379]
            # The correct formula is: L1_d = B1.T @ B1 (not B2.T @ B2)
            L1_d = torch.mm(B1.t(), B1)    # [379 × 379] - edges × edges
            
            # L1_u: Up edge laplacian (edges × edges) - this should also be [379 × 379]
            # The correct formula is: L1_u = B2 @ B2.T (not B1.T @ B1)
            L1_u = torch.mm(B2, B2.t())    # [379 × 379] - edges × edges  
            
            # L2: Face laplacian (triangles × triangles)
            L2 = torch.mm(B2.t(), B2)      # [165 × 165] - triangles × triangles
            
            # Validate laplacian dimensions before proceeding
            if L1_u.shape[0] < 10 or L1_u.shape[1] < 10:
                print(f"Warning: Window {window_data.get('window_idx', 'unknown')} has insufficient edge structure: L1_u={L1_u.shape}")
                continue
            
            if L2.shape[0] < 2 or L2.shape[1] < 2:
                print(f"Warning: Window {window_data.get('window_idx', 'unknown')} has insufficient triangle structure: L2={L2.shape}")
                continue
            
            laplacian_all = (L0, L1_d, L1_u, L2)
            
            # Prepare incidence matrices for SCCNN
            # SCCNN expects: incidence_all = (B1, B2)
            incidence_all = (B1, B2)
            
            # Apply SCCNN: processes all simplex levels together
            output_features = self.sccnn(x_all, laplacian_all, incidence_all)
            
            # SCCNN returns (node_features, edge_features, triangle_features)
            # We'll use edge features for our pipeline (most informative for connectivity)
            node_embeddings, edge_embeddings, triangle_embeddings = output_features
            
            # Global pooling on edge embeddings (most relevant for brain connectivity)
            window_embedding = torch.mean(edge_embeddings, dim=0)
            per_window_vec.append(window_embedding)
        
        # Check if we have any valid windows
        if len(per_window_vec) == 0:
            print("Warning: No valid windows found for this person")
            # Return a zero tensor with correct shape
            device = next(self.parameters()).device
            return torch.zeros(2, device=device)  # [num_classes]
        
        # Stack window embeddings
        window_embeddings = torch.stack(per_window_vec)
        
        # Apply GRU
        gru_out, _ = self.gru(window_embeddings.unsqueeze(0))
        
        # Take last output
        final_embedding = gru_out[0, -1]
        
        # Classification
        logits = self.head(final_embedding)
        
        # Ensure logits has the correct shape [batch_size, num_classes]
        # Since we're processing one person at a time, batch_size=1
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)  # [1, num_classes]
        
        return logits


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Add tqdm progress bar for training batches
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch in progress_bar:
        # Each batch item is a person with multiple windows
        person_data = batch[0]  # batch_size=1, so first item
        label = person_data['label']
        label_tensor = torch.tensor([label], dtype=torch.long).to(device)
        
        # Forward pass - process all windows for this person
        optimizer.zero_grad()
        logits = model(person_data)
        loss = criterion(logits, label_tensor)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        pred = logits.argmax(dim=1)  # logits is [1, 2], so argmax(dim=1) gives [1]
        correct += (pred == label_tensor).sum().item()
        total += 1
        
        # Update progress bar with current metrics
        current_acc = correct / total
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{current_acc:.4f}'
        })
    
    return total_loss / len(train_loader), correct / total


def evaluate(model, test_loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    # Add tqdm progress bar for evaluation batches
    progress_bar = tqdm(test_loader, desc="Evaluating", leave=False)
    
    with torch.no_grad():
        for batch in progress_bar:
            # Each batch item is a person with multiple windows
            person_data = batch[0]  # batch_size=1, so first item
            label = person_data['label']
            label_tensor = torch.tensor([label], dtype=torch.long).to(device)
            
            logits = model(person_data)
            loss = criterion(logits, label_tensor)
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)  # logits is [1, 2], so argmax(dim=1) gives [1]
            correct += (pred == label_tensor).sum().item()
            total += 1
            
            all_preds.append(pred.cpu().numpy())
            all_labels.append(label)
            
            # Update progress bar with current metrics
            current_acc = correct / total
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.4f}'
            })
    
    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)
    
    return avg_loss, accuracy, all_preds, all_labels


def plot_results(history, save_path):
    """Plot training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy (TD)', 'Unhealthy (FEP)'],
                yticklabels=['Healthy (TD)', 'Unhealthy (FEP)'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train SCCNN model for brain connectivity classification')
    parser.add_argument('--data', default='export_tnx_windows', help='Data directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--hidden_edge', type=int, default=64, help='Hidden edge dimension')
    parser.add_argument('--gru_hidden', type=int, default=128, help='GRU hidden dimension')
    parser.add_argument('--test_split', type=float, default=0.2, help='Test split ratio')
    parser.add_argument('--output', default='trained_sccnn_model', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("Starting SCCNN Training")
    print("===================================")
    print(f"Data directory: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Test split: {args.test_split}")
    
    # Load manifest
    manifest_path = os.path.join(args.data, 'manifest.json')
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest not found at {manifest_path}")
        print("Please run export_windows_classification.py first")
        return
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Get unique subjects
    subjects = list(manifest['subjects'].keys())
    print(f"Found {len(subjects)} subjects")
    
    # Split subjects into train/test
    train_subjects, test_subjects = train_test_split(
        subjects, test_size=args.test_split, random_state=42, 
        stratify=[manifest['subjects'][s]['label'] for s in subjects]
    )
    
    print(f"Train subjects: {len(train_subjects)}")
    print(f"Test subjects: {len(test_subjects)}")
    
    # Create datasets
    train_dataset = SubjectDataset(args.data, train_subjects, test_subjects, 'train')
    test_dataset = SubjectDataset(args.data, train_subjects, test_subjects, 'test')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # SCCNN uses fixed channel dimensions, so we don't need dynamic feature extraction
    # All features (nodes, edges, triangles) will have 1 channel input and 64 hidden channels
    
    model = TemporalSCCNNClassifier(
        d0=1, d1=1, d2=1,  # Fixed channel dimensions for SCCNN
        hidden_edge=args.hidden_edge,
        gru_hidden=args.gru_hidden
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    # Training loop
    print("\nStarting training...")
    best_val_acc = 0

    # Add tqdm progress bar for epochs
    epoch_progress = tqdm(range(args.epochs), desc="Epochs")

    for epoch in epoch_progress:
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Update epoch progress bar
        epoch_progress.set_postfix({
            'Train Loss': f'{train_loss:.4f}',
            'Train Acc': f'{train_acc:.4f}',
            'Val Loss': f'{val_loss:.4f}',
            'Val Acc': f'{val_acc:.4f}'
        })
        
        # Print progress (keep the original logging too)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': args
            }, os.path.join(args.output, 'best_model.pth'))
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")
    
    # Load best model for final evaluation
    checkpoint = torch.load(os.path.join(args.output, 'best_model.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_loss, final_acc, all_preds, all_labels = evaluate(model, test_loader, criterion, device)
    
    print(f"Final test accuracy: {final_acc:.4f}")
    
    # Flatten predictions and labels
    all_preds = [p[0] for p in all_preds]
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                              target_names=['Healthy (TD)', 'Unhealthy (FEP)']))
    
    # Save results
    results = {
        'test_accuracy': float(final_acc),  # Convert numpy.float64 to Python float
        'test_loss': float(final_loss),     # Convert numpy.float64 to Python float
        'best_val_accuracy': float(best_val_acc),  # Convert numpy.float64 to Python float
        'predictions': [int(p) for p in all_preds],  # Convert numpy.int64 to Python int
        'true_labels': [int(l) for l in all_labels],  # Convert numpy.int64 to Python int
        'train_subjects': train_subjects,
        'test_subjects': test_subjects,
        'args': vars(args)
    }
    
    with open(os.path.join(args.output, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot results
    plot_results(history, os.path.join(args.output, 'training_history.png'))
    plot_confusion_matrix(all_labels, all_preds, os.path.join(args.output, 'confusion_matrix.png'))
    
    print(f"\nResults saved to {args.output}/")
    print("SCCNN Pipeline completed successfully!")


if __name__ == "__main__":
    main() 