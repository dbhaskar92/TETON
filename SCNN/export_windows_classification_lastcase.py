#!/usr/bin/env python3
"""

"""

import os
import json
import math
import glob
from itertools import combinations
from collections import defaultdict
import heapq

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import pysindy as ps
from joblib import Parallel, delayed

# ------------------ User params (edit as needed) ------------------
EEG_DIR        = '../eeg/'          # Directory containing all EEG files
FS             = 256.0              # [Hz]
WIN_SG, ORDER  = 13, 3              # SavGol window (odd) & polynomial order
WIN_LEN        = 1024               # samples per sliding window
STRIDE         = 512                # hop between windows
D_MAX          = 2                  # 2→ up to triangles; 3→ allow tetrahedra
THRESH_SINDY   = 0.2                # STLSQ sparsity threshold (model-fitting phase)

# Adaptive mapping threshold (per window) for |coef| → simplex
TAU_MODE        = 'percentile'      # 'percentile' or 'mad'
TAU_PCTL        = 97.5              # if TAU_MODE='percentile'
TAU_KMAD        = 6.0               # if TAU_MODE='mad' → tau = median + KMAD * MAD
ENFORCE_CLOSURE = True              # make each window's output a simplicial complex

# SINDy processing limits
MAX_ROWS        = 3000              # cap rows per window during SINDy
N_JOBS          = -1                # joblib cores (-1 = all)

# Locality Analysis Parameters (PC-based)
R_TARGET_PC  = 1.0                 # ~1 z-unit RMS per channel kept
R_DROP95_PC  = 3.5                 # drop only if window is wildly nonlocal
BAD_KEPT_PC  = 2.0                 # if kept subset is still too wide, skip
K_MIN        = 600                 # minimum kept timestamps
K_MAX        = 1000                # optional cap

# Analysis Options
CENTER_METHOD   = 'median'          # 'mean' or 'median' for Taylor expansion center

# Output directory
OUTDIR = 'export_tnx_windows_improved'
os.makedirs(OUTDIR, exist_ok=True)

# ------------------ Label extraction ------------------

def extract_label_from_filename(filename):
    """Extract label from filename: TD = 0 (healthy), FEP = 1 (unhealthy)"""
    if 'TD' in filename:
        return 0  # Healthy
    elif 'FEP' in filename:
        return 1  # Unhealthy
    else:
        raise ValueError(f"Unknown label in filename: {filename}")

def get_all_eeg_files():
    eeg_files = []
    csv_pattern = os.path.join(EEG_DIR, '*_EEGdata.csv')
    csv_files = glob.glob(csv_pattern)
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        label = extract_label_from_filename(filename)
        subject_id = filename.replace('_EEGdata.csv', '')
        eeg_files.append({
            'filepath': csv_file,
            'filename': filename,
            'subject_id': subject_id,
            'label': label,
            'label_name': 'Healthy' if label == 0 else 'Unhealthy'
        })
    return sorted(eeg_files, key=lambda x: x['subject_id'])

# ------------------ Locality Analysis Helpers (PC-based) ------------------

def robust_window_distances(Xw_centered, eps=1e-8):
    med   = np.median(Xw_centered, axis=1, keepdims=True)
    mad   = np.median(np.abs(Xw_centered - med), axis=1, keepdims=True)
    sigma = np.maximum(eps, 1.4826 * mad)
    Z = Xw_centered / sigma
    d_raw = np.sqrt((Z**2).sum(axis=0))     # L2 across channels
    d_pc  = d_raw / np.sqrt(Z.shape[0])     # per-channel RMS z-distance
    return d_pc, sigma

def select_local_indices(Xw):
    x0 = np.median(Xw, axis=1, keepdims=True)
    Xc = Xw - x0
    d_pc, sigma = robust_window_distances(Xc)
    p95_pc = float(np.percentile(d_pc, 95))

    keep = np.where(d_pc <= R_TARGET_PC)[0]
    used = "radius_pc"
    if keep.size < K_MIN:
        keep = np.argsort(d_pc)[:K_MIN]; used = "kmin_pc"
    elif keep.size > K_MAX:
        keep = np.argsort(d_pc)[:K_MAX]; used = "kmax_pc"

    p95_kept_pc = float(np.percentile(d_pc[keep], 95)) if keep.size > 0 else np.nan

    if (p95_pc > R_DROP95_PC) and (p95_kept_pc > BAD_KEPT_PC):
        return {"skip": True, "x0": x0, "sigma": sigma,
                "p95_raw": p95_pc, "p95_kept": p95_kept_pc, "used": "drop_pc"}

    return {"skip": False, "x0": x0, "sigma": sigma, "keep": np.sort(keep),
            "p95_raw": p95_pc, "p95_kept": p95_kept_pc, "used": used}

def preselect_windows(X, WIN_LEN, STRIDE):
    metas = []
    starts = range(0, X.shape[1] - WIN_LEN + 1, STRIDE)
    nW = len(starts)
    print(f"\n--- Performing Robust Locality Pre-selection ---")
    print(f"="*40)
    print(f"Analyzing {nW} windows with WIN_LEN={WIN_LEN}, STRIDE={STRIDE}...")
    print(f"Locality Params: R_TARGET_PC={R_TARGET_PC}, K_MIN={K_MIN}, K_MAX={K_MAX}, "
          f"R_DROP95_PC={R_DROP95_PC}, BAD_KEPT_PC={BAD_KEPT_PC}\n")

    for wi, w_start in enumerate(starts, 1):
        w_end = w_start + WIN_LEN
        Xw = X[:, w_start:w_end]
        out = select_local_indices(Xw)
        t_mid = (w_start + w_end) / 2 / FS

        if out["skip"]:
            print(f"Window {wi}/{nW} (t={t_mid:.3f}s): SKIP ({out['used']}) "
                  f"p95_raw={out['p95_raw']:.2f}, p95_kept={out['p95_kept']:.2f}")
            metas.append({
                "w_start": w_start, "t_mid": t_mid, "skip": True, "keep": [],
                "p95_raw": out["p95_raw"], "p95_kept": out["p95_kept"],
                "kept_pct": 0.0, "used": out["used"]
            })
            continue

        keep = out["keep"]
        kept_pct = 100.0 * keep.size / WIN_LEN
        print(f"Window {wi}/{nW} (t={t_mid:.3f}s): kept={keep.size}/{WIN_LEN} ({kept_pct:.1f}%), "
              f"p95_raw={out['p95_raw']:.2f}, p95_kept={out['p95_kept']:.2f}, mode={out['used']}")

        metas.append({
            "w_start": w_start, "t_mid": t_mid, "skip": False, "keep": keep.tolist(),
            "x0": out["x0"].flatten().tolist(), "sigma": out["sigma"].flatten().tolist(),
            "p95_raw": out["p95_raw"], "p95_kept": out["p95_kept"],
            "kept_pct": float(kept_pct), "used": out["used"]
        })

    kept_counts = [len(m["keep"]) for m in metas if not m["skip"]]
    drops = sum(m["skip"] for m in metas)
    print("\n" + "="*40)
    print("--- Locality Pre-selection Summary ---")
    print("="*40)
    print(f"Total Windows: {nW}   Dropped (nonlocal): {drops}")
    if kept_counts:
        print(f"Points kept per window (min/mean/median/max): "
              f"{min(kept_counts)}/{np.mean(kept_counts):.1f}/{np.median(kept_counts):.1f}/{max(kept_counts)}")

    return metas

# ------------------ SINDy + hyperedge extraction ------------------

def indices_from_term(term_str):
    if term_str == '1':
        return []
    idxs = []
    for tok in term_str.split():
        base, *pow_part = tok.split('^')
        j = int(base[1:])
        power = int(pow_part[0]) if pow_part else 1
        idxs.extend([j] * power)
    return idxs

def _parse_feature_types_via_indices(feature_names):
    kinds, lin_var, cross_pair = [], [], []
    for name in feature_names:
        idxs = [] if name == '1' else indices_from_term(name)
        if len(idxs) == 1:
            kinds.append('lin');  lin_var.append(idxs[0]); cross_pair.append(None)
        elif len(idxs) == 2 and len(set(idxs)) == 2:
            i, j = sorted(idxs)
            kinds.append('cross'); lin_var.append(None);    cross_pair.append((i, j))
        else:
            kinds.append('other'); lin_var.append(None);    cross_pair.append(None)
    return kinds, lin_var, cross_pair

def extract_edge_triangle_scores(A, feature_names, n):
    """Return S2 (n×n) from linear terms and S3 dict for triangles from cross terms."""
    kinds, lin_of, cross_pair = _parse_feature_types_via_indices(feature_names)
    lin_cols   = [c for c,k in enumerate(kinds) if k == 'lin']
    cross_cols = [c for c,k in enumerate(kinds) if k == 'cross']
    absA = np.abs(A)

    # edges: symmetric from linear terms only
    S2_dir = np.zeros((n, n), dtype=float)
    for i in range(n):                # target
        for c in lin_cols:
            j = lin_of[c]
            if j is None or j == i:
                continue
            S2_dir[i, j] = max(S2_dir[i, j], absA[i, c])
    S2 = np.maximum(S2_dir, S2_dir.T) # undirected

    # triangles: genuine cross terms (target i, monomial x_j x_k with j<k, j!=i, k!=i)
    S3 = defaultdict(float)
    for i in range(n):   # target
        for c in cross_cols:
            j, k = cross_pair[c]
            if i in (j, k):  # skip target participation in the monomial
                continue
            key = frozenset((i, j, k))
            if absA[i, c] > S3[key]:
                S3[key] = absA[i, c]
    return S2, S3

def mst_bottleneck_value(S2):
    n = S2.shape[0]
    if n < 2:
        return float('inf')

    # Prim's algorithm for MAX spanning tree
    visited = [False]*n
    visited[0] = True
    heap = []
    for v in range(1, n):
        w = S2[0, v]
        heapq.heappush(heap, (-w, 0, v))  # max-heap via negative weight

    chosen = 0
    mins = []  # collect MST edge weights; we'll take min at the end
    while heap and chosen < n - 1:
        negw, u, v = heapq.heappop(heap)
        if visited[v]:
            continue
        visited[v] = True
        w = -negw
        mins.append(w)
        chosen += 1
        for x in range(n):
            if not visited[x] and x != v:
                heapq.heappush(heap, (-S2[v, x], v, x))

    if chosen != n - 1:
        # graph had zero/very small weights that don't connect -> bottleneck 0
        return 0.0
    return min(mins) if mins else 0.0

def build_complex_from_scores(S2, S3, tau2, tau3, enforce_closure=True):
    """Build closed complex from fixed thresholds tau2 (edges) and tau3 (triangles)."""
    n = S2.shape[0]
    edges = {frozenset((i, j))
             for i in range(n) for j in range(i+1, n)
             if S2[i, j] >= tau2}

    tris  = {t for t, s in S3.items() if s >= tau3}

    if enforce_closure:
        # add triangle faces
        for t in tris:
            i, j, k = sorted(t)
            edges.add(frozenset((i, j)))
            edges.add(frozenset((i, k)))
            edges.add(frozenset((j, k)))

    return edges, tris

# Global polynomial library (linear + interactions up to D_MAX)
GLOBAL_LIBRARY = ps.PolynomialLibrary(
    degree=D_MAX,
    include_bias=False,
    interaction_only=False
)

def _tau_from_abs(absA):
    """Compute per-window tau from |coef| matrix according to TAU_MODE."""
    v = absA.ravel()
    if TAU_MODE.lower() == 'mad':
        med = np.median(v)
        mad = 1.4826 * np.median(np.abs(v - med))
        return float(med + TAU_KMAD * mad)
    # default: percentile
    return float(np.percentile(v, TAU_PCTL))

def fit_window_from_meta(meta, X, Y, dt):
    """Fit SINDy model for a single window."""
    if meta["skip"]:
        return {"t_mid": meta["t_mid"], "skip": True, "reason": "nonlocal window"}

    w_start = meta["w_start"]; w_end = w_start + WIN_LEN
    Xw_orig, Yw_orig = X[:, w_start:w_end], Y[:, w_start:w_end]

    x0   = np.array(meta["x0"]).reshape(-1, 1)
    keep = np.array(meta["keep"])
    if keep.size == 0:
        return {"t_mid": meta["t_mid"], "skip": True, "reason": "empty keep list"}

    Xw_c = Xw_orig - x0
    Xw_use, Yw_use = Xw_c[:, keep], Yw_orig[:, keep]

    if Xw_use.shape[1] > MAX_ROWS:
        idx = np.linspace(0, Xw_use.shape[1]-1, MAX_ROWS, dtype=int)
        Xw_use, Yw_use = Xw_use[:, idx], Yw_use[:, idx]

    try:
        optimizer = ps.STLSQ(alpha=1e-3, threshold=THRESH_SINDY)
        model = ps.SINDy(feature_library=GLOBAL_LIBRARY, optimizer=optimizer)
        model.fit(Xw_use.T, t=dt, x_dot=Yw_use.T, quiet=True)

        A = model.coefficients()
        names = model.get_feature_names()

        # Extract raw scores (no threshold here)
        S2, S3 = extract_edge_triangle_scores(A, names, X.shape[0])
        
        # Print statistics before thresholding
        n_nodes = X.shape[0]
        n_edges_before = np.count_nonzero(S2)
        n_triangles_before = len(S3)
        print(f"      Window {meta['w_start']//STRIDE + 1}: BEFORE thresholding:")
        print(f"        Nodes: {n_nodes}, Edges: {n_edges_before}, Triangles: {n_triangles_before}")
        print(f"        S2 shape: {S2.shape}, S3 keys: {len(S3)}")
        print(f"        S2 non-zero: {np.count_nonzero(S2)}, S2 max: {np.max(S2):.6f}")
        if S3:
            print(f"        S3 max score: {max(S3.values()):.6f}, S3 min score: {min(S3.values()):.6f}")

        return {
            't_mid': meta["t_mid"],
            'S2': S2,                         # (n x n) symmetric edge scores
            'S3': S3,                         # dict of triangle scores
            'x0': x0.flatten().tolist(),
            'n_samples_fit': int(Xw_use.shape[1]),
            'n_samples_kept': len(meta["keep"]),
            'kept_pct': meta["kept_pct"],
            'locality_mode': meta["used"],
            'p95_raw': meta["p95_raw"],
            'p95_kept': meta["p95_kept"],
            'skip': False
        }
    except Exception as e:
        return {"t_mid": meta["t_mid"], "skip": True, "reason": f"SINDy fitting failed: {str(e)}"}

def create_incidence_matrices(edges, triangles, n, oriented_B2=True):
    """
    Create B1 and B2 incidence matrices for TopoNetX.
    Returns:
      (B1_rows, B1_cols, B1_data, (n, m_edges)),
      (B2_rows, B2_cols, B2_data, (m_edges, n_tris)),
      edge_list, triangle_list
    """
    # Deterministic lists (and canonical edge orientation u<v)
    edge_list = sorted([tuple(sorted(e)) for e in edges])
    triangle_list = sorted([tuple(sorted(t)) for t in triangles])

    # B1: nodes to edges  (edge col has +1 at u, -1 at v for u<v)
    B1_rows, B1_cols, B1_data = [], [], []
    for e_idx, (u, v) in enumerate(edge_list):
        B1_rows.extend([u, v])
        B1_cols.extend([e_idx, e_idx])
        B1_data.extend([1, -1])
    
    # B2: edges to triangles
    # If oriented_B2: use boundary ∂(i,j,k) = (j,k) - (i,k) + (i,j),
    # where each edge is oriented (u<v). Otherwise, put +1 for all three.
    edge_index = { (u, v): e_idx for e_idx, (u, v) in enumerate(edge_list) }
    B2_rows, B2_cols, B2_data = [], [], []

    for t_idx, tri in enumerate(triangle_list):
        i, j, k = tri  # sorted order i<j<k
        tri_edges = []
        if oriented_B2:
            # (j,k) with +1
            if (j, k) in edge_index:   e_idx, sgn = edge_index[(j, k)], +1
            else:                      e_idx, sgn = edge_index[(k, j)], -1
            tri_edges.append((e_idx, sgn))
            # (i,k) with -1
            if (i, k) in edge_index:   e_idx, sgn = edge_index[(i, k)], -1
            else:                      e_idx, sgn = edge_index[(k, i)], +1
            tri_edges.append((e_idx, sgn))
            # (i,j) with +1
            if (i, j) in edge_index:   e_idx, sgn = edge_index[(i, j)], +1
            else:                      e_idx, sgn = edge_index[(j, i)], -1
            tri_edges.append((e_idx, sgn))
        else:
            # un-oriented: +1 for each of its three edges
            for (u, v) in [(i, j), (i, k), (j, k)]:
                e_idx = edge_index.get((u, v), edge_index.get((v, u)))
                if e_idx is not None:
                    tri_edges.append((e_idx, +1))

        for e_idx, sgn in tri_edges:
            B2_rows.append(e_idx)
            B2_cols.append(t_idx)
            B2_data.append(int(sgn))
    
    return (B1_rows, B1_cols, B1_data, (n, len(edge_list))), \
           (B2_rows, B2_cols, B2_data, (len(edge_list), len(triangle_list))), \
           edge_list, triangle_list

def process_eeg_file(csv_file, subject_id, label):
    """Process a single EEG file and return window data."""
    print(f"Processing {subject_id} (Label: {label})...")
    
    # Load EEG data
    try:
        raw = pd.read_csv(csv_file, header=None, usecols=range(31)).values.astype(np.float64)
        print(f"  Loaded {raw.shape[0]} timepoints × {raw.shape[1]} channels")
    except Exception as e:
        print(f"  Error loading {csv_file}: {e}")
        return None
    
    # Drop constant/all-zero channels
    tol = 1e-12
    nz_mask = (np.ptp(raw, axis=0) > tol)
    if (~nz_mask).any():
        dropped = np.where(~nz_mask)[0] + 1
        print(f"  ⚠️  Dropping constant channels (1-based): {dropped.tolist()}")
    raw = raw[:, nz_mask]
    raw = raw.T                       # rows = channels, cols = time
    n, T_raw = raw.shape
    dt = 1 / FS
    
    print(f"  Data after preprocessing: {n} channels × {T_raw} samples ({T_raw/FS:.1f} seconds)")
    
    # Savitzky-Golay smooth + derivative
    half = (WIN_SG - 1) // 2
    X_smooth_full = savgol_filter(raw, WIN_SG, ORDER, axis=1, mode='interp')
    dXdt_full     = savgol_filter(raw, WIN_SG, ORDER, deriv=1, delta=dt, axis=1, mode='interp')
    
    # Trim ends equally to reduce edge artifacts
    X_smooth = X_smooth_full[:, half:-half]
    dXdt_raw = dXdt_full[:,       half:-half]
    
    # Per-channel Normalization
    print("  Applying per-channel normalization...")
    eps = 1e-8
    mu  = X_smooth.mean(axis=1, keepdims=True)
    sig = X_smooth.std(axis=1, keepdims=True) + eps
    
    X = (X_smooth - mu) / sig
    Y = dXdt_raw / sig  # scale only
    
    print("  Normalization complete.")
    
    # Preselect windows
    window_meta = preselect_windows(X, WIN_LEN, STRIDE)
    
    # Fit SINDy over selected windows
    fit_metas = [m for m in window_meta if not m["skip"]]
    skipped_count = len(window_meta) - len(fit_metas)
    print(f"\n  --- Proceeding to SINDy Fitting ---")
    print(f"  Fitting {len(fit_metas)} windows (Skipped {skipped_count} nonlocal windows)")
    
    if len(fit_metas) == 0:
        print("  No windows to fit!")
        return []
    
    # Fit windows (sequential for now to avoid complexity)
    results = []
    for meta in fit_metas:
        result = fit_window_from_meta(meta, X, Y, dt)
        results.append(result)
    
    # Keep only successfully fitted windows
    successful_results = [res for res in results if not res.get("skip", False)]
    fit_skipped_count = len(results) - len(successful_results)
    print(f"  Successfully fitted {len(successful_results)} windows (Skipped {fit_skipped_count} during fit)")
    
    if len(successful_results) == 0:
        print("  No successful SINDy fits!")
        return []
    
    # Compute global thresholds
    bottlenecks = []
    for res in successful_results:
        b = mst_bottleneck_value(res['S2'])
        bottlenecks.append(b)
    tau2_global = float(min(bottlenecks)) if bottlenecks else 0.0
    
    all_tri_scores = []
    for res in successful_results:
        all_tri_scores.extend(list(res['S3'].values()))
    if all_tri_scores:
        tau3_global = float(np.quantile(all_tri_scores, 0.85))  # Changed from 0.975 to 0.85
    else:
        tau3_global = float('inf')
    
    print(f"  Global thresholds: tau2={tau2_global:.6g}, tau3={tau3_global:.6g}")
    
    # Build per-window complexes
    windows = []
    for res in successful_results:
        E2, T3 = build_complex_from_scores(res['S2'], res['S3'], tau2_global, tau3_global, enforce_closure=True)
        
        # Print statistics after thresholding
        n_edges_after = len(E2)
        n_triangles_after = len(T3)
        print(f"      Window {int(res['t_mid'] * FS) // STRIDE + 1}: AFTER thresholding:")
        print(f"        Edges: {n_edges_after}, Triangles: {n_triangles_after}")
        print(f"        Thresholds: tau2={tau2_global:.6f}, tau3={tau3_global:.6f}")
        
        if len(E2) > 0:
            # Create incidence matrices (also get the deterministic lists)
            B1_info, B2_info, edge_list, triangle_list = create_incidence_matrices(E2, T3, n, oriented_B2=True)
            
            # Print incidence matrix statistics
            B1_shape = B1_info[3]
            B2_shape = B2_info[3]
            print(f"        Incidence matrices:")
            print(f"          B1 shape: {B1_shape} (nodes × edges)")
            print(f"          B2 shape: {B2_shape} (edges × triangles)")
            print(f"          B1 non-zero elements: {len(B1_info[0])}")
            print(f"          B2 non-zero elements: {len(B2_info[0])}")
            
            # ----- Cochain features -----
            # H0 at exact window midpoint (rounded & clamped)
            mid_idx = int(round(res['t_mid'] * FS))
            mid_idx = max(0, min(mid_idx, X.shape[1]-1))
            H0 = X[:, mid_idx].astype(np.float32)

            # H1 from S2 strengths aligned with edge_list
            H1_vals = []
            for (u, v) in edge_list:
                H1_vals.append(float(res['S2'][u, v]))
            H1 = np.array(H1_vals, dtype=np.float32)[:, None]

            # H2 from S3 strengths aligned with triangle_list
            H2_vals = []
            for tri in triangle_list:
                H2_vals.append(float(res['S3'].get(frozenset(tri), 0.0)))
            H2 = np.array(H2_vals, dtype=np.float32)[:, None]
            # --------------------------------

            # Print feature statistics
            print(f"        Features:")
            print(f"          H0 shape: {H0.shape} (node features)")
            print(f"          H1 shape: {H1.shape} (edge features)  | min/max=({H1.min():.4g}, {H1.max():.4g})")
            if H2.size > 0:
                print(f"          H2 shape: {H2.shape} (triangle features) | min/max=({H2.min():.4g}, {H2.max():.4g})")
            else:
                print(f"          H2 shape: {H2.shape} (triangle features) | no triangles")
            print(f"          H0 range: [{H0.min():.3f}, {H0.max():.3f}]")
            print(f"        " + "-"*50)
            
            window_info = {
                'subject_id': subject_id,
                'label': label,
                'window_idx': len(windows),
                't_mid': res['t_mid'],
                'B1': B1_info,
                'B2': B2_info,
                'H0': H0,
                'H1': H1,
                'H2': H2,
                'n_edges': len(E2),
                'n_triangles': len(T3),
                'tau2_global': tau2_global,
                'tau3_global': tau3_global,
                'locality_stats': {
                    'n_samples_fit': res['n_samples_fit'],
                    'n_samples_kept': res['n_samples_kept'],
                    'kept_pct': res['kept_pct'],
                    'p95_raw': res['p95_raw'],
                    'p95_kept': res['p95_kept'],
                    'mode': res['locality_mode']
                }
            }
            
            windows.append(window_info)
    
    print(f"  Generated {len(windows)} valid windows")
    return windows

def main():
    print("Starting Pipeline")
    print("==================================")
    
    # Get all EEG files
    eeg_files = get_all_eeg_files()
    print(f"Found {len(eeg_files)} EEG files:")
    
    for eeg_file in eeg_files:
        print(f"  {eeg_file['subject_id']}: {eeg_file['label_name']} (Label {eeg_file['label']})")
    
    print("\nProcessing files...")
    
    # Process each file sequentially (more reliable for SINDy)
    all_windows = []
    subject_stats = {}
    
    for i, eeg_file in enumerate(eeg_files):
        print(f"\n[{i+1}/{len(eeg_files)}] Processing {eeg_file['subject_id']}...")
        windows = process_eeg_file(eeg_file['filepath'], eeg_file['subject_id'], eeg_file['label'])
        
        if windows:
            all_windows.extend(windows)
            subject_stats[eeg_file['subject_id']] = {
                'label': eeg_file['label'],
                'n_windows': len(windows)
            }
            print(f"   {eeg_file['subject_id']}: {len(windows)} windows processed")
        else:
            print(f"   {eeg_file['subject_id']}: No valid windows generated")
    
    if not all_windows:
        print("No valid windows generated. Exiting.")
        return
    
    print(f"\nTotal windows generated: {len(all_windows)}")
    
    # Save windows and manifest
    print("Saving windows...")
    
    manifest = {
        'subjects': subject_stats,
        'total_windows': len(all_windows),
        'window_size': WIN_LEN,
        'stride': STRIDE,
        'fs': FS,
        'parameters': {
            'win_sg': WIN_SG,
            'order': ORDER,
            'd_max': D_MAX,
            'thresh_sindy': THRESH_SINDY,
            'tau_mode': TAU_MODE,
            'tau_pctl': TAU_PCTL,
            'tau_kmad': TAU_KMAD,
            'enforce_closure': ENFORCE_CLOSURE,
            'r_target_pc': R_TARGET_PC,
            'r_drop95_pc': R_DROP95_PC,
            'bad_kept_pc': BAD_KEPT_PC,
            'k_min': K_MIN,
            'k_max': K_MAX,
            'center_method': CENTER_METHOD
        },
        'windows': []
    }
    
    for i, window in enumerate(all_windows):
        filename = f"window_{i:08d}.npz"
        filepath = os.path.join(OUTDIR, filename)
        
        # Save window data
        np.savez_compressed(
            filepath,
            subject_id=window['subject_id'],
            label=window['label'],
            window_idx=window['window_idx'],
            t_mid=window['t_mid'],
            B1_rows=np.array(window['B1'][0]),
            B1_cols=np.array(window['B1'][1]),
            B1_data=np.array(window['B1'][2]),
            B1_shape=np.array(window['B1'][3]),
            B2_rows=np.array(window['B2'][0]),
            B2_cols=np.array(window['B2'][1]),
            B2_data=np.array(window['B2'][2]),
            B2_shape=np.array(window['B2'][3]),
            H0=window['H0'],
            H1=window['H1'],
            H2=window['H2'],
            n_edges=window['n_edges'],
            n_triangles=window['n_triangles'],
            tau2_global=window['tau2_global'],
            tau3_global=window['tau3_global'],
            locality_stats=window['locality_stats']
        )
        
        manifest['windows'].append({
            'file': filename,
            'subject_id': window['subject_id'],
            'label': window['label'],
            'window_idx': window['window_idx'],
            't_mid': window['t_mid'],
            'n_edges': window['n_edges'],
            'n_triangles': window['n_triangles']
        })
    
    # Save manifest
    manifest_path = os.path.join(OUTDIR, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Manifest saved to {manifest_path}")
    
    # Print summary
    print("\nPipeline Summary:")
    print("=================")
    print(f"Total subjects: {len(subject_stats)}")
    print(f"Total windows: {len(all_windows)}")
    
    label_counts = defaultdict(int)
    for window in all_windows:
        label_counts[window['label']] += 1
    
    print(f"Windows per label:")
    print(f"  Label 0 (Healthy): {label_counts[0]}")
    print(f"  Label 1 (Unhealthy): {label_counts[1]}")
    
    print(f"\nOutput directory: {OUTDIR}")
    print("Pipeline completed successfully")

if __name__ == "__main__":
    main()
