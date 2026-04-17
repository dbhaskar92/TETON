#!/usr/bin/env python3
"""
Load and print the contents of one .npz subject file from abide_sindy_processed/subjects/.
"""

import numpy as np
import os

SUBJECTS_DIR = os.path.join(os.path.dirname(__file__), "abide_sindy_processed", "subjects")


def print_subject(filepath: str) -> None:
    data = np.load(filepath, allow_pickle=True)

    print(f"File: {os.path.basename(filepath)}")
    print(f"  label:      {data['label']}")
    print(f"  subject_id: {data['subject_id']}")
    print(f"  n_windows:  {data['n_windows']}")

    windows = data["windows"]
    print(f"  windows:    {len(windows)} windows\n")

    for i, window in enumerate(windows):
        print(f"  Window {i}:")
        for key, val in window.items():
            if isinstance(val, np.ndarray):
                print(f"    {key:20s} shape={val.shape}  dtype={val.dtype}")
            else:
                print(f"    {key:20s} {val}")
        print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Default: first file in the subjects directory
        files = sorted(f for f in os.listdir(SUBJECTS_DIR) if f.endswith(".npz"))
        if not files:
            print(f"No .npz files found in {SUBJECTS_DIR}")
            sys.exit(1)
        filepath = os.path.join(SUBJECTS_DIR, files[0])

    print_subject(filepath)
