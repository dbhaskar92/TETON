#!/usr/bin/env python3
"""
EEG Dataset Processing Script

This script processes EEG data files and creates a dataset with:
- Windowed EEG samples from event segments
- Labels in (2, 4) format: [group encoding, event type encoding]
- SINDy topological analysis for edge/triangle detection

Usage:
    python process_eeg_dataset.py [--data_dir DATA_DIR] [--output_dir OUTPUT_DIR]
                                  [--window_length WINDOW_LENGTH] [--stride STRIDE]
                                  [--run_sindy] [--sampling_rate SAMPLING_RATE]
"""

import pandas as pd
import numpy as np
import os
import re
import json
import pickle
import argparse
from glob import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    'data_dir': 'data',
    'output_dir': 'processed_eeg_dataset',
    'window_length': 768,
    'stride': 384,
    'sampling_rate': 256,  # Hz
    'run_sindy': False,
}


# ============================================================================
# File Parsing Functions
# ============================================================================

def parse_eeg_filename(filename):
    """
    Parse EEG filename to extract group, participant number, and file type.
    Format: EEG_{FEP/TD}_{participant_no}_EEG{data/event/xyz}.csv
    """
    pattern = r'EEG_(FEP|TD)_(\d+)_EEG(data|event|xyz)\.csv'
    match = re.match(pattern, filename)
    if match:
        return {
            'group': match.group(1),
            'participant': match.group(2),
            'file_type': match.group(3)
        }
    return None


def discover_participants(data_dir):
    """Discover and organize all EEG files by participant."""
    all_files = os.listdir(data_dir)
    eeg_files = [f for f in all_files if f.startswith('EEG_') and f.endswith('.csv')]
    
    participants = {}
    for filename in eeg_files:
        parsed = parse_eeg_filename(filename)
        if parsed:
            key = f"{parsed['group']}_{parsed['participant']}"
            if key not in participants:
                participants[key] = {
                    'group': parsed['group'],
                    'participant': parsed['participant'],
                    'data_file': None,
                    'event_file': None,
                    'xyz_file': None
                }
            participants[key][f"{parsed['file_type']}_file"] = os.path.join(data_dir, filename)
    
    # Filter to complete participants (must have data + event files)
    complete = {k: v for k, v in participants.items() 
                if v['data_file'] and v['event_file']}
    
    return complete


# ============================================================================
# Label Encoding Functions
# ============================================================================

def encode_group(group):
    """Encode group as one-hot vector [FEP, TD]."""
    return np.array([1, 0]) if group == 'FEP' else np.array([0, 1])


def encode_event_type(event_type):
    """Encode event type as one-hot vector [Type1, Type2, Type3, Type4]."""
    encoding = np.zeros(4)
    if 1 <= event_type <= 4:
        encoding[event_type - 1] = 1
    return encoding


def create_label(group, event_type):
    """
    Create (2, 4) label matrix:
    Row 0: [group_FEP, group_TD, 0, 0] (padded to 4)
    Row 1: [event_1, event_2, event_3, event_4]
    """
    label = np.zeros((2, 4))
    label[0, :2] = encode_group(group)
    label[1, :] = encode_event_type(event_type)
    return label


# ============================================================================
# Window Extraction Functions
# ============================================================================

def extract_windows_from_segment(data_segment, window_length=768, stride=384):
    """Extract sliding windows from a data segment."""
    windows = []
    num_samples = data_segment.shape[0]
    
    start = 0
    while start + window_length <= num_samples:
        window = data_segment[start:start + window_length]
        windows.append(window)
        start += stride
    
    return windows


def get_event_segments(event_data, total_samples):
    """Define segments for each event based on event timestamps."""
    segments = []
    event_data = event_data.sort_values('Sample_Index').reset_index(drop=True)
    
    for i in range(len(event_data)):
        start_idx = event_data.loc[i, 'Sample_Index']
        event_type = event_data.loc[i, 'Event_Type']
        
        if i < len(event_data) - 1:
            end_idx = event_data.loc[i + 1, 'Sample_Index']
        else:
            end_idx = total_samples
        
        segments.append((start_idx, end_idx, event_type))
    
    return segments


def process_participant(participant_info, window_length=768, stride=384):
    """
    Process a single participant's data and extract samples.
    
    Returns:
        samples: list of numpy arrays, each (num_windows, window_length, num_channels)
        labels: list of numpy arrays, each (2, 4)
        metadata: list of dicts with participant info
    """
    samples_list = []
    labels_list = []
    metadata_list = []
    
    group = participant_info['group']
    participant = participant_info['participant']
    
    # Load data
    eeg_data = pd.read_csv(participant_info['data_file'], header=None).values
    event_data = pd.read_csv(participant_info['event_file'], header=None)
    event_data.columns = ['Sample_Index', 'Event_Type']
    
    total_samples = eeg_data.shape[0]
    segments = get_event_segments(event_data, total_samples)
    
    for start_idx, end_idx, event_type in segments:
        segment = eeg_data[start_idx:end_idx]
        
        if segment.shape[0] < window_length:
            continue
        
        windows = extract_windows_from_segment(segment, window_length, stride)
        
        if len(windows) == 0:
            continue
        
        sample_windows = np.stack(windows, axis=0)
        label = create_label(group, event_type)
        
        samples_list.append(sample_windows)
        labels_list.append(label)
        metadata_list.append({
            'group': group,
            'participant': participant,
            'event_type': event_type,
            'segment_start': start_idx,
            'segment_end': end_idx,
            'num_windows': len(windows)
        })
    
    return samples_list, labels_list, metadata_list


# ============================================================================
# SINDy Analysis Functions
# ============================================================================

def run_sindy_analysis(all_samples, all_metadata, config):
    """Run SINDy topological analysis on all samples (per-window, not aggregated)."""
    try:
        from SINDy import SINDy_EEG_sample_analysis
        from argparse import Namespace
    except ImportError:
        print("Warning: SINDy module not found. Skipping SINDy analysis.")
        return None, None
    
    dt = 1.0 / config['sampling_rate']
    
    sindy_args = Namespace(
        win_sg=15,
        order=3,
        half=7,
        dt=dt,
        d_max=2,
        rho_val=0.5,
        admm_rho=3.0,
        admm_overrelax=1.6,
        max_iters=300,
        q=0.7,
        S3_mode='mean',
        pair_lambda_mult=2.0,
        row_norm_nnz_thr=1e-4,
        ROW_NORM_NNZ_THR=1e-4,
        PARAM_ABS_NNZ_THR=1e-6,
        r_target_pc=3.0,
        k_min=50,
        k_max=500,
        max_rows=1000,
        win_len=config['window_length'],
        scale=1.0,
    )
    
    sindy_results = []
    
    print("Processing samples with SINDy analysis (per-window)...")
    for sample_idx in tqdm(range(len(all_samples)), desc="SINDy Analysis"):
        sample_windows = all_samples[sample_idx]
        
        try:
            # Get per-window edge/triangle predictions (aggregate=False)
            window_results = SINDy_EEG_sample_analysis(sample_windows, sindy_args, aggregate=False)
            
            # window_results is a list of dicts, one per window:
            # {'edge_sweep': set, 'tri_sweep': set, 'window_idx': int}
            sindy_results.append({
                'sample_idx': sample_idx,
                'num_windows': len(window_results),
                'window_results': window_results
            })
        except Exception as e:
            print(f"  Sample {sample_idx}: ERROR - {str(e)}")
            sindy_results.append({
                'sample_idx': sample_idx,
                'num_windows': 0,
                'window_results': [],
                'error': str(e)
            })
    
    return sindy_results, sindy_args


# ============================================================================
# Main Processing Pipeline
# ============================================================================

def process_dataset(config):
    """Main function to process the entire EEG dataset."""
    
    print("=" * 70)
    print("EEG DATASET PROCESSING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Data Directory: {config['data_dir']}")
    print(f"  Output Directory: {config['output_dir']}")
    print(f"  Window Length: {config['window_length']}")
    print(f"  Stride: {config['stride']}")
    print(f"  Run SINDy: {config['run_sindy']}")
    
    # Discover participants
    print("\n" + "=" * 70)
    print("Step 1: Discovering EEG files...")
    complete_participants = discover_participants(config['data_dir'])
    
    fep_count = sum(1 for p in complete_participants.values() if p['group'] == 'FEP')
    td_count = sum(1 for p in complete_participants.values() if p['group'] == 'TD')
    print(f"  Found {len(complete_participants)} complete participants (FEP: {fep_count}, TD: {td_count})")
    
    # Process all participants
    print("\n" + "=" * 70)
    print("Step 2: Processing participants...")
    
    all_samples = []
    all_labels = []
    all_metadata = []
    
    for key, info in tqdm(sorted(complete_participants.items()), desc="Participants"):
        try:
            samples, labels, metadata = process_participant(
                info, config['window_length'], config['stride']
            )
            all_samples.extend(samples)
            all_labels.extend(labels)
            all_metadata.extend(metadata)
        except Exception as e:
            print(f"  {key}: ERROR - {str(e)}")
    
    print(f"\n  Total samples extracted: {len(all_samples)}")
    
    # Analyze window statistics
    window_counts = [s.shape[0] for s in all_samples]
    print(f"  Windows per sample: min={min(window_counts)}, max={max(window_counts)}, mean={np.mean(window_counts):.1f}")
    
    # Convert labels to array
    y = np.array(all_labels, dtype=np.float32)
    X = all_samples  # Keep as list
    
    # Run SINDy analysis if requested
    sindy_results = None
    sindy_args = None
    if config['run_sindy']:
        print("\n" + "=" * 70)
        print("Step 3: Running SINDy topological analysis...")
        sindy_results, sindy_args = run_sindy_analysis(all_samples, all_metadata, config)
    
    # Save dataset
    print("\n" + "=" * 70)
    print("Step 4: Saving dataset...")
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Save samples as pickle (preserves list of variable-sized arrays)
    with open(os.path.join(config['output_dir'], 'X_samples.pkl'), 'wb') as f:
        pickle.dump(X, f)
    
    # Save labels as numpy array
    np.save(os.path.join(config['output_dir'], 'y_labels.npy'), y)
    
    # Save metadata
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(os.path.join(config['output_dir'], 'metadata.csv'), index=False)
    
    # Save dataset info
    dataset_info = {
        'num_samples': len(X),
        'total_windows': sum(s.shape[0] for s in X),
        'window_length': config['window_length'],
        'stride': config['stride'],
        'num_channels': X[0].shape[2] if X else 0,
        'y_shape': list(y.shape),
        'num_participants': len(complete_participants),
        'fep_count': fep_count,
        'td_count': td_count,
        'window_counts': {
            'min': int(min(window_counts)),
            'max': int(max(window_counts)),
            'mean': float(np.mean(window_counts))
        }
    }
    
    with open(os.path.join(config['output_dir'], 'dataset_info.json'), 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # Save SINDy results if available (per-window structure)
    if sindy_results is not None:
        # Structure: List of samples, each sample is a list of window results
        # sindy_edges_per_sample[sample_idx][window_idx] = list of edges
        # sindy_triangles_per_sample[sample_idx][window_idx] = list of triangles
        
        sindy_edges_per_sample = []
        sindy_triangles_per_sample = []
        window_metadata = []
        
        for r in sindy_results:
            sample_idx = r['sample_idx']
            sample_edges = []
            sample_triangles = []
            
            for w_result in r['window_results']:
                # Convert frozensets to sorted lists for storage
                edges = [sorted(list(e)) for e in w_result['edge_sweep']]
                triangles = [sorted(list(t)) for t in w_result['tri_sweep']]
                sample_edges.append(edges)
                sample_triangles.append(triangles)
                
                # Collect per-window metadata
                window_metadata.append({
                    'sample_idx': sample_idx,
                    'window_idx': w_result['window_idx'],
                    'num_edges': len(w_result['edge_sweep']),
                    'num_triangles': len(w_result['tri_sweep']),
                    'group': all_metadata[sample_idx]['group'],
                    'event_type': all_metadata[sample_idx]['event_type']
                })
            
            sindy_edges_per_sample.append(sample_edges)
            sindy_triangles_per_sample.append(sample_triangles)
        
        # Create per-window DataFrame
        sindy_window_df = pd.DataFrame(window_metadata)
        
        sindy_save_data = {
            'sindy_results': sindy_results,
            'sindy_edges_per_sample': sindy_edges_per_sample,  # [sample][window] = list of edges
            'sindy_triangles_per_sample': sindy_triangles_per_sample,  # [sample][window] = list of triangles
            'sindy_args': vars(sindy_args) if sindy_args else {},
            'sindy_window_df': sindy_window_df
        }
        
        with open(os.path.join(config['output_dir'], 'sindy_topology.pkl'), 'wb') as f:
            pickle.dump(sindy_save_data, f)
        
        # Save per-window summary as CSV
        sindy_window_df.to_csv(os.path.join(config['output_dir'], 'sindy_window_summary.csv'), index=False)
        
        # Print SINDy statistics
        print(f"\n  SINDy Analysis Statistics (per-window):")
        print(f"    Total windows analyzed: {len(window_metadata)}")
        if window_metadata:
            edge_counts = [w['num_edges'] for w in window_metadata]
            triangle_counts = [w['num_triangles'] for w in window_metadata]
            print(f"    Edges per window: min={min(edge_counts)}, max={max(edge_counts)}, mean={np.mean(edge_counts):.1f}")
            print(f"    Triangles per window: min={min(triangle_counts)}, max={max(triangle_counts)}, mean={np.mean(triangle_counts):.1f}")
    
    # Print summary
    print(f"\nDataset saved to '{config['output_dir']}/'")
    print("\nFiles created:")
    for f in os.listdir(config['output_dir']):
        filepath = os.path.join(config['output_dir'], f)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  {f}: {size_mb:.2f} MB")
    
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    
    return X, y, metadata_df, sindy_results


# ============================================================================
# Command Line Interface
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Process EEG dataset and create windowed samples with optional SINDy analysis.'
    )
    parser.add_argument('--data_dir', type=str, default=DEFAULT_CONFIG['data_dir'],
                        help='Directory containing EEG CSV files')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_CONFIG['output_dir'],
                        help='Output directory for processed dataset')
    parser.add_argument('--window_length', type=int, default=DEFAULT_CONFIG['window_length'],
                        help='Length of each window in samples')
    parser.add_argument('--stride', type=int, default=DEFAULT_CONFIG['stride'],
                        help='Stride between consecutive windows')
    parser.add_argument('--sampling_rate', type=int, default=DEFAULT_CONFIG['sampling_rate'],
                        help='EEG sampling rate in Hz')
    parser.add_argument('--run_sindy', action='store_true',
                        help='Run SINDy topological analysis')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    config = {
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'window_length': args.window_length,
        'stride': args.stride,
        'sampling_rate': args.sampling_rate,
        'run_sindy': args.run_sindy,
    }
    
    X, y, metadata_df, sindy_results = process_dataset(config)
