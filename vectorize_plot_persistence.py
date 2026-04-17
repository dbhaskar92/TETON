#!/usr/bin/env python3
"""
Vectorize zigzag persistence diagrams (H0, H1) and project to 2D with PCA
(or UMAP if available) to visualise class separation.

Each subject must have a pre-computed *_zigzag_dgms.npz file produced by
zigzag_persistence.py.  The original *.npz file is used to retrieve the
class label.

Vectorization
-------------
  --method images  : persistence images via persim  (default if persim installed)
  --method stats   : 6-dim statistical summary       (always available, fallback)
  --method betti   : Betti curves sampled on a uniform grid

Dimensionality reduction
------------------------
  --reducer pca    : PCA  (default, always available)
  --reducer umap   : UMAP (requires umap-learn)

Usage
-----
    python vectorize_plot_persistence.py
    python vectorize_plot_persistence.py --method stats --reducer umap
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SUBJECTS_DIR = os.path.join(os.path.dirname(__file__),
                            "abide_sindy_processed_slow")
                            #"abide_sindy_processed", "subjects")

# ── colour / label config ────────────────────────────────────────────────────
CLASS_COLORS = {0: "#0072B2", 1: "#E69F00", -1: "#999999"}
CLASS_NAMES  = {0: "Class 0 (Control)", 1: "Class 1 (ASD)", -1: "Unknown"}


# ── data loading ─────────────────────────────────────────────────────────────

def load_diagrams(path: str):
    """Return (H0, H1) arrays from a *_zigzag_dgms.npz file."""
    d = np.load(path)
    return d["H0"].astype(np.float32), d["H1"].astype(np.float32)


def load_label(source_path: str):
    """Return the integer label from a .npz or .json source file, or None if absent."""
    if source_path.endswith(".json"):
        import json
        with open(source_path) as f:
            d = json.load(f)
        val = d.get("label")
        return int(val) if val is not None else None
    d = np.load(source_path, allow_pickle=True)
    val = d["label"]
    return int(val) if val is not None else None


def find_source_file(dgm_path: str):
    """
    Given a *_zigzag_dgms.npz path, return the path of the matching source
    file (.npz or .json), or None if neither exists.
    """
    base = dgm_path.replace("_zigzag_dgms.npz", "")
    for ext in (".npz", ".json"):
        candidate = base + ext
        if os.path.exists(candidate):
            return candidate
    return None


def drop_inf(pts: np.ndarray):
    """Remove rows with infinite death; return finite points and the max death seen."""
    if len(pts) == 0:
        return pts, 1.0
    finite = pts[np.isfinite(pts[:, 1])]
    max_death = float(finite[:, 1].max()) if len(finite) > 0 else 1.0
    return finite, max_death


# ── vectorization methods ────────────────────────────────────────────────────

def vectorize_persistence_images(diagrams: list, resolution: int = 20) -> np.ndarray:
    """
    Vectorize a list of (N,2) birth/death arrays using persistence images.

    Ranges are derived from data upfront so the imager is constructed with
    explicit birth_range / pers_range / pixel_size — this avoids the slow
    auto-fitting that hangs on large zigzag diagrams.

    Returns shape (n_subjects, resolution*resolution).
    """
    from persim import PersistenceImager

    # Convert to (birth, lifetime) format; drop infinite deaths
    bl_list = []
    all_births, all_lifetimes = [], []
    for pts in diagrams:
        finite, _ = drop_inf(pts)
        if len(finite) == 0:
            bl_list.append(np.empty((0, 2), dtype=np.float32))
        else:
            bl = np.column_stack([finite[:, 0],
                                  finite[:, 1] - finite[:, 0]]).astype(np.float32)
            bl_list.append(bl)
            all_births.append(bl[:, 0])
            all_lifetimes.append(bl[:, 1])

    if not all_births:
        raise ValueError("All diagrams are empty – cannot build persistence images.")

    b_all = np.concatenate(all_births)
    l_all = np.concatenate(all_lifetimes)
    b_min, b_max = float(b_all.min()), float(b_all.max())
    l_max = float(l_all.max())

    # pixel_size chosen so the grid is exactly (resolution × resolution)
    pixel_size = max((b_max - b_min) / resolution, l_max / resolution, 1e-3)

    pimgr = PersistenceImager(
        birth_range=(b_min, b_max),
        pers_range=(0.0, l_max),
        pixel_size=pixel_size,
    )

    n_pixels = int(np.prod(pimgr.resolution))
    features = []
    for bl in bl_list:
        if len(bl) == 0:
            features.append(np.zeros(n_pixels, dtype=np.float32))
        else:
            img = pimgr.transform([bl])[0]
            features.append(img.flatten().astype(np.float32))
    return np.array(features, dtype=np.float32)


def vectorize_stats(diagrams: list) -> np.ndarray:
    """
    6-dimensional statistical summary per diagram:
      [n_bars, sum_lifetime, mean_lifetime, std_lifetime, max_lifetime, mean_birth]
    Essential (infinite) bars are excluded from lifetime statistics but counted.
    """
    features = []
    for pts in diagrams:
        finite, _ = drop_inf(pts)
        n_inf = int(np.sum(~np.isfinite(pts[:, 1]))) if len(pts) > 0 else 0
        if len(finite) == 0:
            features.append([0, 0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        lt = finite[:, 1] - finite[:, 0]
        features.append([
            len(finite) + n_inf,        # total bars (finite + essential)
            float(lt.sum()),
            float(lt.mean()),
            float(lt.std()) if len(lt) > 1 else 0.0,
            float(lt.max()),
            float(finite[:, 0].mean()),  # mean birth
        ])
    return np.array(features, dtype=np.float32)


def vectorize_betti(diagrams: list, n_grid: int = 50) -> np.ndarray:
    """
    Betti curves sampled on a uniform grid of n_grid points covering the
    global [min_birth, max_death] range.  Each point counts how many bars
    are alive at that filtration value.
    """
    # Find global range
    all_births, all_deaths = [], []
    for pts in diagrams:
        if len(pts) == 0:
            continue
        finite, _ = drop_inf(pts)
        if len(finite) > 0:
            all_births.append(finite[:, 0].min())
            all_deaths.append(finite[:, 1].max())
    if not all_births:
        return np.zeros((len(diagrams), n_grid), dtype=np.float32)

    t_min, t_max = min(all_births), max(all_deaths)
    grid = np.linspace(t_min, t_max, n_grid)

    features = []
    for pts in diagrams:
        finite, _ = drop_inf(pts)
        curve = np.zeros(n_grid, dtype=np.float32)
        for t_idx, t in enumerate(grid):
            if len(finite) > 0:
                curve[t_idx] = float(np.sum((finite[:, 0] <= t) & (finite[:, 1] >= t)))
        features.append(curve)
    return np.array(features, dtype=np.float32)


# ── dimensionality reduction ──────────────────────────────────────────────────

def reduce_pca(X: np.ndarray):
    X_s = StandardScaler().fit_transform(X)
    # Zero-variance columns produce NaN after scaling (divide by 0); zero them out.
    X_s = np.nan_to_num(X_s, nan=0.0, posinf=0.0, neginf=0.0)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X_s)
    var = pca.explained_variance_ratio_
    axis_labels = (f"PC1 ({var[0]*100:.1f}%)", f"PC2 ({var[1]*100:.1f}%)")
    return Z, axis_labels


def reduce_umap(X: np.ndarray):
    import umap
    X_s = StandardScaler().fit_transform(X)
    X_s = np.nan_to_num(X_s, nan=0.0, posinf=0.0, neginf=0.0)
    reducer = umap.UMAP(n_components=2, random_state=42)
    Z = reducer.fit_transform(X_s)
    return Z, ("UMAP-1", "UMAP-2")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Vectorize zigzag persistence diagrams → PCA / UMAP scatter plot")
    parser.add_argument("--subjects_dir", default=SUBJECTS_DIR,
                        help="Directory containing *_zigzag_dgms.npz files")
    parser.add_argument("--method", choices=["auto", "images", "stats", "betti"],
                        default="auto",
                        help="Vectorization method (default: auto → images if persim available)")
    parser.add_argument("--reducer", choices=["pca", "umap"], default="pca",
                        help="Dimensionality reduction method (default: pca)")
    parser.add_argument("--resolution", type=int, default=20,
                        help="Persistence image resolution (default: 20)")
    parser.add_argument("--betti_grid", type=int, default=50,
                        help="Number of Betti-curve grid points (default: 50)")
    args = parser.parse_args()

    # ── discover files ────────────────────────────────────────────────────────
    dgm_files = sorted(glob.glob(os.path.join(args.subjects_dir, "*_zigzag_dgms.npz")))
    if not dgm_files:
        print(f"[ERROR] No *_zigzag_dgms.npz files found in {args.subjects_dir}")
        print("        Run zigzag_persistence.py on each subject first.")
        sys.exit(1)

    H0_list, H1_list, labels, loaded = [], [], [], []
    for path in dgm_files:
        source = find_source_file(path)
        if source is None:
            print(f"[WARN] Source file not found, skipping: {path}")
            continue
        try:
            H0, H1 = load_diagrams(path)
            label   = load_label(source)
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")
            continue
        H0_list.append(H0)
        H1_list.append(H1)
        labels.append(label)
        loaded.append(os.path.basename(path))

    if not H0_list:
        print("[ERROR] No valid subjects loaded.")
        sys.exit(1)

    has_labels = any(l is not None for l in labels)
    if has_labels:
        # Replace any missing labels with -1 so numpy array stays integer
        labels = np.array([l if l is not None else -1 for l in labels])
        n0, n1 = int((labels == 0).sum()), int((labels == 1).sum())
        print(f"Loaded {len(labels)} subjects: {n0} class-0, {n1} class-1")
    else:
        labels = np.zeros(len(H0_list), dtype=int)
        print(f"Loaded {len(labels)} subjects (no labels available – plotted as single class)")

    # ── choose vectorization method ───────────────────────────────────────────
    method = args.method
    if method == "auto":
        try:
            import persim  # noqa: F401
            method = "images"
        except ImportError:
            method = "stats"
    print(f"Vectorization method : {method}")

    if method == "images":
        print("  Fitting persistence imager …")
        feat_H0 = vectorize_persistence_images(H0_list, resolution=args.resolution)
        feat_H1 = vectorize_persistence_images(H1_list, resolution=args.resolution)
    elif method == "betti":
        feat_H0 = vectorize_betti(H0_list, n_grid=args.betti_grid)
        feat_H1 = vectorize_betti(H1_list, n_grid=args.betti_grid)
    else:  # stats
        feat_H0 = vectorize_stats(H0_list)
        feat_H1 = vectorize_stats(H1_list)

    # Concatenate H0 and H1 features
    X = np.hstack([feat_H0, feat_H1])
    print(f"Feature matrix       : {X.shape[0]} subjects × {X.shape[1]} dims")

    # ── dimensionality reduction ──────────────────────────────────────────────
    reducer_name = args.reducer
    print(f"Dimensionality reduction: {reducer_name.upper()}")
    if reducer_name == "umap":
        try:
            Z, axis_labels = reduce_umap(X)
        except ImportError:
            print("[WARN] umap-learn not installed; falling back to PCA.")
            reducer_name = "pca"
            Z, axis_labels = reduce_pca(X)
    else:
        Z, axis_labels = reduce_pca(X)

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left panel: combined H0 + H1 features
    ax = axes[0]
    for cls in sorted(np.unique(labels)):
        mask = labels == cls
        ax.scatter(Z[mask, 0], Z[mask, 1],
                   c=CLASS_COLORS.get(int(cls), "grey"),
                   label=CLASS_NAMES.get(int(cls), f"Class {cls}"),
                   alpha=0.8, edgecolors="white", linewidths=0.3, s=55)
    ax.set_xlabel(axis_labels[0]); ax.set_ylabel(axis_labels[1])
    ax.set_title(f"{reducer_name.upper()} – H0 + H1 features\n({method})")
    ax.legend(framealpha=0.85)

    # Right panel: H0 and H1 separately (PCA on each, coloured by class)
    ax2 = axes[1]
    for feat, dim_label, marker in [(feat_H0, "H0", "o"), (feat_H1, "H1", "^")]:
        if feat.shape[1] < 2:
            continue
        Z_dim, al = reduce_pca(feat)
        for cls in sorted(np.unique(labels)):
            mask = labels == cls
            ax2.scatter(Z_dim[mask, 0], Z_dim[mask, 1],
                        c=CLASS_COLORS.get(int(cls), "grey"),
                        marker=marker, alpha=0.7,
                        edgecolors="white", linewidths=0.3, s=45,
                        label=f"{CLASS_NAMES.get(int(cls), f'Class {cls}')} – {dim_label}")
    ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
    ax2.set_title(f"PCA per homology dimension\n({method})\n○=H0  △=H1")
    ax2.legend(framealpha=0.85, fontsize=7)

    fig.suptitle("Zigzag persistence vectorization – class separation", fontsize=13)
    plt.tight_layout()

    out = os.path.join(args.subjects_dir, "plots/pca_persistence.png")
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
