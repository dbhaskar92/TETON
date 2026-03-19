# ============================================================
# ABIDE → DMN ROIs (Harvard-Oxford) → SINDy Simplicial Complexes
# One ASD subject, one control, per-window complexes compared
# ============================================================

import numpy as np
import types
from scipy.signal import savgol_filter
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from SINDy import (
    precompute_index_patterns, build_maps_from_patterns,
    build_soc_edges_from_maps, degree_weights, SINDy_windowed
)
import pickle
import json
from pathlib import Path

# ==============================================================
# 1. Harvard-Oxford atlas + DMN ROI selection
# ==============================================================
ho_atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
ho_labels = ho_atlas.labels   # list of strings, index 0 = Background

# Core DMN regions in Harvard-Oxford cortical atlas (by label substring)
DMN_KEYWORDS = [
    'Frontal Medial Cortex',
    'Precuneous Cortex',          # note HO spells it this way
    'Cingulate Gyrus, posterior',
    'Angular Gyrus',
    'Middle Frontal Gyrus',
    'Inferior Frontal Gyrus',     # pars triangularis / orbitalis
    'Parahippocampal Gyrus',
    'Temporal Pole',
    'Lateral Occipital Cortex',   # posterior DMN
]

# Find which 1-based atlas label indices match DMN keywords
# (NiftiLabelsMasker uses 1-based label values matching atlas voxel integers)
dmn_indices = []   # 1-based, matching atlas integer values
dmn_names   = []
for idx, label in enumerate(ho_labels):
    for kw in DMN_KEYWORDS:
        if kw.lower() in label.lower():
            dmn_indices.append(idx)   # ho_labels[0] = 'Background' → atlas value 1 = ho_labels[1]
            dmn_names.append(label)
            break

print(f"DMN ROIs selected ({len(dmn_indices)}):")
for i, name in zip(dmn_indices, dmn_names):
    print(f"  [{i}] {name}")

# ==============================================================
# 2. Fetch ABIDE: 1 ASD + 1 Control, using func_preproc NIfTI
# ==============================================================
abide_asd = datasets.fetch_abide_pcp(
    n_subjects=1,
    pipeline='cpac',
    band_pass_filtering=True,
    global_signal_regression=False,
    derivatives=['func_preproc'],
    quality_checked=True,
    DX_GROUP=1,      # 1 = Autism
    verbose=1
)

abide_ctrl = datasets.fetch_abide_pcp(
    n_subjects=1,
    pipeline='cpac',
    band_pass_filtering=True,
    global_signal_regression=False,
    derivatives=['func_preproc'],
    quality_checked=True,
    DX_GROUP=2,      # 2 = Control
    verbose=1
)

print(f"\nASD subject:     {abide_asd.phenotypic['FILE_ID'].values[0]}")
print(f"Control subject: {abide_ctrl.phenotypic['FILE_ID'].values[0]}")

# ==============================================================
# 3. Extract DMN time series via NiftiLabelsMasker
# ==============================================================
# We restrict the masker to only the DMN label values
masker = NiftiLabelsMasker(
    labels_img=ho_atlas.maps,
    labels=ho_labels,
    mask_img=None,
    standardize='zscore_sample',
    memory='nilearn_cache',
    verbose=0
)

def extract_dmn_signals(func_file, masker, dmn_indices):
    """
    Extract and return only the DMN columns from the full HO time series.
    Returns array of shape (T, n_dmn_rois).
    """
    # Full HO signals: shape (T, n_ho_rois)
    full_signals = masker.fit_transform(func_file)
    # dmn_indices are 0-based into ho_labels (which includes Background at 0)
    # NiftiLabelsMasker outputs one column per non-background label in order,
    # so column j corresponds to ho_labels[j+1]. We therefore subtract 1.
    col_indices = [i - 1 for i in dmn_indices if i > 0]
    # guard against out-of-range
    col_indices = [c for c in col_indices if c < full_signals.shape[1]]
    return full_signals[:, col_indices]

ts_asd  = extract_dmn_signals(abide_asd.func_preproc[0],  masker, dmn_indices)
ts_ctrl = extract_dmn_signals(abide_ctrl.func_preproc[0], masker, dmn_indices)

print(f"\nASD  time series shape:  {ts_asd.shape}")   # (T, n_dmn)
print(f"Ctrl time series shape:  {ts_ctrl.shape}")

# ==============================================================
# 4. SINDy args
# ==============================================================
# fMRI TR is typically 2.0s for ABIDE (check phenotypic if unsure)
TR = float(abide_asd.phenotypic.get('TR', [2.0])[0]) if 'TR' in abide_asd.phenotypic.columns else 2.0

WIN_SG = 11   # Savitzky-Golay window (must be odd, < win_len)
HALF   = (WIN_SG - 1) // 2

args = types.SimpleNamespace(
    # Library
    d_max            = 3,        # 2: edges only, 3: also triangles
    win_len          = 40,      # ~80s window at TR=2s
    stride           = 10,      # 20s stride
    max_rows         = 300,
    k_min            = 20,
    k_max            = 200,
    r_target_pc      = 2.0,

    # Savitzky-Golay
    win_sg           = WIN_SG,
    order            = 3,
    dt               = TR,
    half             = HALF,

    # ADMM
    scale            = 1.0,
    rho_val          = 1.0,
    admm_rho         = 3.0,
    admm_overrelax   = 1.6,
    max_iters        = 150,

    # Thresholding
    q                = 0.75,
    S3_mode          = 'mean',
    pair_lambda_mult = 2.0,

    # Sparsification
    ROW_NORM_NNZ_THR  = 1e-4,
    PARAM_ABS_NNZ_THR = 1e-5,
    row_norm_nnz_thr  = 1e-4,

    feature_aggr      = 'mean',
)

# ==============================================================
# 5. Core SINDy preprocessing (shared setup)
# ==============================================================
def preprocess_timeseries(ts, args):
    """
    ts: (T, N) numpy array — already standardized by masker
    Returns X (N, T'), Y (N, T') ready for SINDy_windowed.
    """
    raw = ts.T  # (N, T)
    # Drop near-constant channels (shouldn't happen post-masker but safety check)
    raw = raw[np.ptp(raw, axis=1) > 1e-12, :]
    n, T = raw.shape

    X_smooth = savgol_filter(raw, args.win_sg, args.order, axis=1, mode='interp')[:, args.half:-args.half]
    dXdt_raw = savgol_filter(raw, args.win_sg, args.order, deriv=1,
                             delta=args.dt, axis=1, mode='interp')[:, args.half:-args.half]
    mu  = X_smooth.mean(axis=1, keepdims=True)
    sig = X_smooth.std(axis=1, keepdims=True) + 1e-8
    X   = (X_smooth - mu) / sig
    Y   = dXdt_raw / sig
    return X, Y, n

def build_sindy_structures(n, args):
    idx_patterns   = precompute_index_patterns(n, args.d_max, use_xp=True)
    maps0, g_total = build_maps_from_patterns(n, args.d_max, idx_patterns)
    edges0         = build_soc_edges_from_maps(args.d_max, maps0)
    w_degree_base  = degree_weights(maps0, g_total, args)
    return idx_patterns, maps0, edges0, w_degree_base

def compute_simplicial_complexes(ts, args, label="subject"):
    """
    Returns a list of dicts, one per sliding window:
      { 'window': i, 't_start': s, 'edges': set of frozensets,
        'triangles': set of frozensets }
    """
    X, Y, n = preprocess_timeseries(ts, args)
    T_trimmed = X.shape[1]
    idx_patterns, maps0, edges0, w_degree_base = build_sindy_structures(n, args)

    complexes = []
    win_idx = 0
    for w_start in range(0, T_trimmed - args.win_len + 1, args.stride):
        w_end = w_start + args.win_len
        Xw = X[:, w_start:w_end]
        Yw = Y[:, w_start:w_end]

        edges, triangles = SINDy_windowed(
            Xw, Yw, n, idx_patterns, maps0, edges0, w_degree_base, args
        )

        complexes.append({
            'window':    win_idx,
            't_start_s': w_start * args.dt,
            'edges':     edges,
            'triangles': triangles,
        })
        win_idx += 1

    print(f"\n[{label}] {len(complexes)} windows | "
          f"n_nodes={n} | T={T_trimmed} samples")
    return complexes

# ==============================================================
# 6. Run for ASD and Control
# ==============================================================
complexes_asd  = compute_simplicial_complexes(ts_asd,  args, label="ASD")
complexes_ctrl = compute_simplicial_complexes(ts_ctrl, args, label="Control")

# ==============================================================
# 7. Summary comparison
# ==============================================================
def summarize_complexes(complexes, dmn_names, label):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    all_edges = {}
    all_tris  = {}
    for c in complexes:
        for e in c['edges']:
            all_edges[e] = all_edges.get(e, 0) + 1
        for t in c['triangles']:
            all_tris[t]  = all_tris.get(t, 0) + 1

    n_win = len(complexes)
    print(f"  Windows: {n_win}")
    print(f"  Unique edges found:     {len(all_edges)}")
    print(f"  Unique triangles found: {len(all_tris)}")

    print(f"\n  Most persistent edges (appearing in most windows):")
    for edge, cnt in sorted(all_edges.items(), key=lambda x: -x[1])[:8]:
        nodes = sorted(edge)
        names = [dmn_names[i] if i < len(dmn_names) else str(i) for i in nodes]
        print(f"    {names[0]}  ↔  {names[1]}  [{cnt}/{n_win} windows]")

    if all_tris:
        print(f"\n  Most persistent triangles:")
        for tri, cnt in sorted(all_tris.items(), key=lambda x: -x[1])[:5]:
            nodes = sorted(tri)
            names = [dmn_names[i] if i < len(dmn_names) else str(i) for i in nodes]
            print(f"    {' — '.join(names)}  [{cnt}/{n_win} windows]")
    else:
        print(f"\n  No triangles found (try d_max=3 for higher-order interactions)")

    return all_edges, all_tris

edges_asd,  tris_asd  = summarize_complexes(complexes_asd,  dmn_names, "ASD")
edges_ctrl, tris_ctrl = summarize_complexes(complexes_ctrl, dmn_names, "Control")

# ==============================================================
# 8. Edge-level comparison
# ==============================================================
print(f"\n{'='*55}")
print("  Edge comparison (persistence = fraction of windows)")
print(f"{'='*55}")
n_win_asd  = len(complexes_asd)
n_win_ctrl = len(complexes_ctrl)

all_edges_union = set(edges_asd.keys()) | set(edges_ctrl.keys())
print(f"  {'Edge':<50}  {'ASD':>6}  {'Ctrl':>6}")
print(f"  {'-'*50}  {'------':>6}  {'------':>6}")
for edge in sorted(all_edges_union, key=lambda e: -edges_asd.get(e, 0)):
    nodes = sorted(edge)
    names = [dmn_names[i] if i < len(dmn_names) else str(i) for i in nodes]
    label_str = f"{names[0]} ↔ {names[1]}"[:50]
    p_asd  = edges_asd.get(edge,  0) / n_win_asd
    p_ctrl = edges_ctrl.get(edge, 0) / n_win_ctrl
    print(f"  {label_str:<50}  {p_asd:>6.2f}  {p_ctrl:>6.2f}")


# ==============================================================
# 9.Save results
# ==============================================================

def save_complexes(complexes, filepath):
    """
    Save complexes to disk. Uses pickle to preserve frozensets natively.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(complexes, f)
    print(f"Saved {len(complexes)} windows → {filepath}")

def save_complexes_json(complexes, filepath):
    """
    JSON version — more portable/readable, converts frozensets to sorted lists.
    """
    serializable = []
    for c in complexes:
        serializable.append({
            'window':    c['window'],
            't_start_s': c['t_start_s'],
            'edges':     [sorted(list(e)) for e in c['edges']],
            'triangles': [sorted(list(t)) for t in c['triangles']],
        })
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved {len(complexes)} windows → {filepath}")

complexes_asd  = compute_simplicial_complexes(ts_asd,  args, label="ASD")
complexes_ctrl = compute_simplicial_complexes(ts_ctrl, args, label="Control")

# Save both formats
save_complexes(complexes_asd,  'results_abide/complexes_asd.pkl')
save_complexes(complexes_ctrl, 'results_abide/complexes_ctrl.pkl')

# JSON as a human-readable backup
save_complexes_json(complexes_asd,  'results_abide/complexes_asd.json')
save_complexes_json(complexes_ctrl, 'results_abide/complexes_ctrl.json')

# Also save the metadata you'll need in the viz script
metadata = {
    'dmn_indices':  dmn_indices,
    'dmn_names':    dmn_names,
    'n_nodes':      len(dmn_names),
    'subject_asd':  str(abide_asd.phenotypic['FILE_ID'].values[0]),
    'subject_ctrl': str(abide_ctrl.phenotypic['FILE_ID'].values[0]),
    'site_asd':     str(abide_asd.phenotypic['SITE_ID'].values[0]),
    'site_ctrl':    str(abide_ctrl.phenotypic['SITE_ID'].values[0]),
    'TR':           float(TR),
    'args': {       # save the args you used so the viz script knows them
        'd_max':   args.d_max,
        'win_len': args.win_len,
        'stride':  args.stride,
        'q':       args.q,
        'S3_mode': args.S3_mode,
        'dt':      args.dt,
    }
}
with open('results_abide/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("Saved metadata.json")