#!/usr/bin/env python3
"""Download the public label files used in examples.md, plus the competition train.csv.

Nothing is redistributed by this repository: every file comes straight from Kaggle under its own
licence. Requirements:
  * pip install kaggle   and an API token in ~/.kaggle/kaggle.json (or KAGGLE_USERNAME/KAGGLE_KEY)
  * You must have joined the competition and accepted its rules on kaggle.com for train.csv.

Usage:
  python scripts/download_public_labels.py --out data
"""
import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

DATASETS = [
    # (kind, source, file inside source, licence / note)
    ("dataset", "nartaa/rsna-knee-hpo-assets", "labels_v1_blend.parquet", "CC0"),
    ("dataset", "riadmohamed42/jev-knee-labels", "HYBRID_labels_scores.csv", "CC0"),
    ("dataset", "nartaa/rsna-knee-s6fall-weights", "training_code/labels/labels_v1_blendjev.parquet",
     "reference copy to check the rebuild; weights dataset is CC BY-NC 4.0, code Apache-2.0"),
    ("competition", "rsna-knee-abnormality-detection", "train.csv", "competition data, rules apply; do not redistribute"),
]


def fetch(kind, source, name, out):
    cmd = ["kaggle", "datasets" if kind == "dataset" else "competitions", "download"]
    cmd += [source] if kind == "dataset" else ["-c", source]
    cmd += ["-f", name, "-p", str(out)]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    base = Path(name).name
    archive = out / (base + ".zip")
    if archive.exists():  # Kaggle zips some single-file downloads
        with zipfile.ZipFile(archive) as z:
            z.extractall(out)
        archive.unlink()
    if not (out / base).exists():
        sys.exit(f"Expected {out / base} after download")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for kind, source, name, note in DATASETS:
        print(f"\n# {source} :: {name}  ({note})")
        fetch(kind, source, name, args.out)
    print(f"\nDone. Files are in {args.out.resolve()}")


if __name__ == "__main__":
    main()
