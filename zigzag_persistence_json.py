#!/usr/bin/env python3
"""
Compute zigzag persistence over the temporal sequence of simplicial complexes
stored in a .json sample file from abide_sindy_processed_slow/.

For a sample with windows K_0, K_1, ..., K_{n-1}, we compute zigzag
persistence over the diagram:

    K_0 ↪ (K_0 ∪ K_1) ↩ K_1 ↪ (K_1 ∪ K_2) ↩ K_2 ↪ ... ↩ K_{n-1}

The zigzag has 2*n - 1 integer time slots:
    even time 2i   → K_i
    odd  time 2i+1 → K_i ∪ K_{i+1}

A simplex is present at:
    - time 2i   if it belongs to K_i
    - time 2i+1 if it belongs to K_i OR K_{i+1}

Dependencies
------------
    pip install dionysus   (requires a C++ compiler; provides zigzag_homology_persistence)
    pip install numpy

Usage
-----
    python zigzag_persistence_json.py                              # first sample file
    python zigzag_persistence_json.py path/to/sample.json         # specific file
"""

import json
import os
import sys

import numpy as np

# dionysus 2.x  -- install with:  pip install dionysus
import dionysus


DATA_DIR = os.path.join(os.path.dirname(__file__), "abide_sindy_processed_slow")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sample(filepath: str):
    """Return (windows, label, sample_id) from a .json file."""
    with open(filepath) as f:
        data = json.load(f)
    windows = data["windows"]
    label = data.get("label")          # may be None if not present
    sample_id = str(data["sample_id"])
    return windows, label, sample_id


def extract_simplices(window: dict) -> set:
    """
    Return all simplices in one window as sorted tuples of vertex indices.
    Includes 0-simplices (nodes), 1-simplices (edges), 2-simplices (triangles).
    """
    simplices = set()

    # 0-simplices – every listed node
    for v in window["nodes"]:
        simplices.add((int(v),))

    # 1-simplices
    for e in window["edges"]:
        simplices.add(tuple(sorted(int(x) for x in e)))

    # 2-simplices
    for t in window["triangles"]:
        if len(t) == 3:
            simplices.add(tuple(sorted(int(x) for x in t)))

    return simplices


# ---------------------------------------------------------------------------
# Zigzag time-interval construction
# ---------------------------------------------------------------------------

def compute_simplex_intervals(window_set: set, n_windows: int) -> list:
    """
    Return the list of closed integer intervals [b, d] during which a simplex
    is present in the zigzag sequence.

    Parameters
    ----------
    window_set : set of int
        Window indices where the simplex is present.
    n_windows : int
        Total number of windows.

    Returns
    -------
    list of float – interleaved [b0, d0, b1, d1, …] as required by
    dionysus.zigzag_homology_persistence (Sequence[SupportsFloat] per simplex).
    """
    n_times = 2 * n_windows - 1
    presence = [False] * n_times

    for i in range(n_windows):
        # Even slot: the simplex must belong to K_i
        if i in window_set:
            presence[2 * i] = True

        # Odd slot: the simplex belongs to K_i ∪ K_{i+1}
        if i < n_windows - 1:
            if i in window_set or (i + 1) in window_set:
                presence[2 * i + 1] = True

    # Compress consecutive True values into closed intervals, stored flat:
    # [birth0, death0, birth1, death1, ...]
    flat = []
    start = None
    for t, here in enumerate(presence):
        if here and start is None:
            start = t
        elif not here and start is not None:
            flat.extend([float(start), float(t - 1)])
            start = None
    if start is not None:
        flat.extend([float(start), float(n_times - 1)])

    return flat


# ---------------------------------------------------------------------------
# Filtration builder
# ---------------------------------------------------------------------------

def build_zigzag_filtration(windows):
    """
    Build a dionysus Filtration and the corresponding times list.

    Returns
    -------
    f : dionysus.Filtration
    times : list[list[float]]
        For each simplex in *f*, a flat [b0, d0, b1, d1, …] list of intervals.
    """
    n_windows = len(windows)

    # Map each simplex to the set of windows it appears in
    simplex_windows: dict = {}
    for i, window in enumerate(windows):
        for s in extract_simplices(window):
            simplex_windows.setdefault(s, set()).add(i)

    # Sort: lower dimension first, then lexicographic – guarantees faces precede cofaces
    ordered = sorted(simplex_windows.keys(), key=lambda s: (len(s), s))

    f = dionysus.Filtration([dionysus.Simplex(list(s)) for s in ordered])
    times = [compute_simplex_intervals(simplex_windows[s], n_windows) for s in ordered]

    return f, times


# ---------------------------------------------------------------------------
# Persistence computation
# ---------------------------------------------------------------------------

def compute_zigzag_persistence(windows, prime: int = 2):
    """
    Compute zigzag persistence diagrams for a sequence of simplicial-complex windows.

    Parameters
    ----------
    windows : list of dict
    prime   : coefficient field characteristic (default 2 → Z/2Z)

    Returns
    -------
    dgms : list[dionysus.Diagram]
        One diagram per homology dimension (index = dimension).
    """
    f, times = build_zigzag_filtration(windows)
    _zz, dgms, _cells = dionysus.zigzag_homology_persistence(f, times, prime=prime)
    return dgms


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_diagrams(dgms, sample_id: str = ""):
    """Pretty-print the persistence diagrams."""
    title = f"Zigzag persistence – {sample_id}" if sample_id else "Zigzag persistence"
    print(title)
    print("=" * len(title))
    for dim, dgm in enumerate(dgms):
        finite = [(p.birth, p.death) for p in dgm if p.death != float("inf")]
        infinite = [(p.birth, p.death) for p in dgm if p.death == float("inf")]
        print(f"\n  H{dim}: {len(finite)} finite bars, {len(infinite)} essential bars")
        if finite:
            print(f"  {'birth':>8}  {'death':>8}  {'length':>8}  notes")
            for b, d in sorted(finite, key=lambda x: -(x[1] - x[0]))[:25]:
                print(f"  {b:>8}  {d:>8}  {d - b:>8}  "
                      f"(windows ≈ {b/2:.1f} – {d/2:.1f})")
            if len(finite) > 25:
                print(f"  ... ({len(finite) - 25} more)")
        if infinite:
            print(f"  Infinite bars (birth only): {[b for b, _ in infinite]}")
    print()


def save_diagrams(dgms, out_path: str):
    """Save diagrams as a .npz file with arrays H0, H1, H2."""
    arrays = {}
    for dim, dgm in enumerate(dgms):
        pts = np.array([[p.birth, p.death] for p in dgm], dtype=np.float32)
        arrays[f"H{dim}"] = pts if len(pts) > 0 else np.empty((0, 2), dtype=np.float32)
    np.savez_compressed(out_path, **arrays)
    print(f"Diagrams saved → {out_path}")


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_sample(filepath: str, skip_existing: bool = True) -> bool:
    """
    Compute and save zigzag persistence for one sample file.

    Returns True on success, False if skipped or failed.
    """
    stem = os.path.splitext(filepath)[0]
    out_path = f"{stem}_zigzag_dgms.npz"

    if skip_existing and os.path.exists(out_path):
        return False   # already done

    try:
        windows, label, sample_id = load_sample(filepath)
        dgms = compute_zigzag_persistence(windows)
        save_diagrams(dgms, out_path)
        return True
    except Exception as exc:
        print(f"  [ERROR] {os.path.basename(filepath)}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import random

    parser = argparse.ArgumentParser(
        description="Compute zigzag persistence for one sample or a batch (.json files).")
    parser.add_argument("filepath", nargs="?", default=None,
                        help="Single .json sample file (omit for batch mode).")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of samples to process in batch mode (default: 100).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for subsampling (default: 42).")
    parser.add_argument("--no_skip", action="store_true",
                        help="Recompute even if *_zigzag_dgms.npz already exists.")
    args = parser.parse_args()

    # ── single-file mode ──────────────────────────────────────────────────────
    if args.filepath is not None:
        print(f"Loading  {args.filepath}")
        windows, label, sample_id = load_sample(args.filepath)
        print(f"  sample_id={sample_id}  label={label}  n_windows={len(windows)}\n")
        print("Building zigzag filtration …")
        print("Computing zigzag persistence …")
        dgms = compute_zigzag_persistence(windows)
        print_diagrams(dgms, sample_id=sample_id)
        stem = os.path.splitext(args.filepath)[0]
        save_diagrams(dgms, f"{stem}_zigzag_dgms.npz")
        sys.exit(0)

    # ── batch mode ────────────────────────────────────────────────────────────
    all_files = sorted(
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith(".json") and not f.endswith("_zigzag_dgms.json")
    )
    if not all_files:
        print(f"No sample .json files found in {DATA_DIR}")
        sys.exit(1)

    rng = random.Random(args.seed)
    sample = rng.sample(all_files, min(args.n_samples, len(all_files)))

    n_total   = len(sample)
    n_done    = 0
    n_skipped = 0
    n_failed  = 0

    print(f"Batch mode: {n_total} samples (seed={args.seed}, "
          f"skip_existing={not args.no_skip})\n")

    for idx, filepath in enumerate(sample, 1):
        sample_name = os.path.basename(filepath)
        stem = os.path.splitext(filepath)[0]
        out_path = f"{stem}_zigzag_dgms.npz"

        if (not args.no_skip) and os.path.exists(out_path):
            print(f"[{idx:>3}/{n_total}] SKIP  {sample_name}")
            n_skipped += 1
            continue

        print(f"[{idx:>3}/{n_total}] Processing {sample_name} …", end=" ", flush=True)
        try:
            windows, label, sample_id = load_sample(filepath)
            dgms = compute_zigzag_persistence(windows)
            save_diagrams(dgms, out_path)
            print(f"done  (label={label}, windows={len(windows)})")
            n_done += 1
        except Exception as exc:
            print(f"FAILED – {exc}")
            n_failed += 1

    print(f"\nSummary: {n_done} computed, {n_skipped} skipped, {n_failed} failed "
          f"(out of {n_total} sampled)")
