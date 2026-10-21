#!/usr/bin/env python3
"""Cell-wise mean of two soft-label tables, keeping table A where table B is missing.

This re-implements the fusion rule nartaa used to build labels_v1_blendjev
(blendjev = mean(labels_v1_blend, JEV HYBRID); where the JEV cell is NaN the blend value is kept).
Original method and reference file: https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights

The row set and order follow table A. Reads .csv or .parquet (parquet needs pyarrow).

Usage:
  python scripts/build_blend.py data/labels_v1_blend.parquet data/HYBRID_labels_scores.csv out/my_blendjev.csv \
      [--reference data/labels_v1_blendjev.parquet]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ID = "StudyInstanceUID"
FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
            "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype={ID: str})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path, help="primary table (rows and order are kept)")
    ap.add_argument("b", type=Path, help="second table; NaN cells fall back to A")
    ap.add_argument("out", type=Path, help="output .csv or .parquet")
    ap.add_argument("--findings", nargs="+", default=FINDINGS)
    ap.add_argument("--reference", type=Path, help="optional published table to compare against")
    args = ap.parse_args()

    A = read(args.a)
    B = read(args.b).drop_duplicates(ID).set_index(ID)
    a = A[args.findings].to_numpy(np.float64)
    b = B.reindex(A[ID])[args.findings].to_numpy(np.float64)
    out = A[[ID]].copy()
    out[args.findings] = np.where(np.isnan(b), a, (a + b) / 2).astype(np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".parquet":
        out.to_parquet(args.out, index=False)
    else:
        out.to_csv(args.out, index=False)
    print(f"rows {len(out)}; cells taken from A only (B missing): {int(np.isnan(b).sum())}")

    if args.reference:
        ref = read(args.reference).set_index(ID).loc[out[ID], args.findings].to_numpy(np.float64)
        print("max |rebuild - reference| = %.3g" % np.abs(out[args.findings].to_numpy(np.float64) - ref).max())


if __name__ == "__main__":
    main()
