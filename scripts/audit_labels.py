#!/usr/bin/env python3
"""Audit two soft-label tables against each other and against report keywords.

For every finding it reports:
  * Pearson correlation between the two tables
  * how many cells differ by more than --threshold, and in which direction
  * how many reports contain a finding keyword while a table scores it low (likely misses)
and writes the suspicious cells to a CSV for human review (it never edits labels).

Keyword matching is a screening tool, not a labeller: it does not handle negation
("no synovitis"), so every flagged cell still needs a human or careful re-read.

Usage (toy data, runs in seconds):
  python scripts/audit_labels.py toy_data/toy_labels_a.csv toy_data/toy_labels_b.csv \
      --reports toy_data/toy_reports.csv --names a b --out audit_out

Usage (RSNA Knee example):
  python scripts/audit_labels.py data/labels_v1_blendjev.parquet my_labels.csv \
      --reports data/train.csv --names blendjev mine --out audit_out
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ID = "StudyInstanceUID"
FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
            "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# Screening patterns (English / Spanish / Portuguese / German). Extend for your data.
KEYWORDS = {
    "ACL": r"\bACL\b|anterior cruciate|cruzado anterior|\bLCA\b|vorderes kreuzband",
    "MCL": r"\bMCL\b|medial collateral|colateral (?:interno|medial)|\bLCM\b|innenband",
    "Medial Meniscus": r"medial menisc|menisco (?:interno|medial)|innenmeniskus",
    "Lateral Meniscus": r"lateral menisc|menisco (?:externo|lateral)|au(?:ss|ß)enmeniskus",
    "Effusion": r"effusion|derrame|erguss",
    "Synovitis": r"s[iy]novi|synovial thicken",
    "Baker's": r"baker|popliteal cyst|quiste popl[ií]teo",
    "Contusion": r"contusion|bone bruise|marrow edema|marrow oedema|contusi[oó]n|edema [óo]seo|knochenmarks[öo]dem",
    "Fracture": r"fracture|fractura|fraktur",
}


def read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype={ID: str})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--reports", type=Path, help="CSV with StudyInstanceUID and a report text column")
    ap.add_argument("--report-col", default="Report")
    ap.add_argument("--names", nargs=2, default=["a", "b"])
    ap.add_argument("--threshold", type=float, default=0.3, help="disagreement threshold")
    ap.add_argument("--low", type=float, default=0.5, help="'scored low' cutoff for keyword misses")
    ap.add_argument("--out", type=Path, default=Path("audit_out"))
    args = ap.parse_args()
    na, nb = args.names

    A, B = read(args.a), read(args.b)
    findings = [f for f in FINDINGS if f in A.columns and f in B.columns]
    m = A[[ID] + findings].merge(B[[ID] + findings], on=ID, suffixes=(f"_{na}", f"_{nb}"))
    if args.reports:
        R = read(args.reports)[[ID, args.report_col]]
        m = m.merge(R, on=ID, how="left")
    print(f"studies in both tables: {len(m)}  (only in {na}: {len(set(A[ID]) - set(m[ID]))}, "
          f"only in {nb}: {len(set(B[ID]) - set(m[ID]))})\n")

    rows, flagged = [], []
    for f in findings:
        x, y = m[f"{f}_{na}"].astype(float), m[f"{f}_{nb}"].astype(float)
        d = x - y
        big = d.abs() > args.threshold
        row = dict(finding=f, pearson=round(float(np.corrcoef(x, y)[0, 1]), 3) if len(m) > 2 else None,
                   mean_a=round(float(x.mean()), 3), mean_b=round(float(y.mean()), 3),
                   disagree=int(big.sum()), a_higher=int((d > args.threshold).sum()),
                   b_higher=int((d < -args.threshold).sum()))
        kw_hit = None
        if args.reports and f in KEYWORDS:
            kw_hit = m[args.report_col].fillna("").str.contains(KEYWORDS[f], flags=re.I, regex=True)
            row["keyword_reports"] = int(kw_hit.sum())
            row[f"kw_but_{na}_low"] = int((kw_hit & (x < args.low)).sum())
            row[f"kw_but_{nb}_low"] = int((kw_hit & (y < args.low)).sum())
            row["disagree_with_kw"] = int((big & kw_hit).sum())
        rows.append(row)
        sel = big if kw_hit is None else big | (kw_hit & ((x < args.low) | (y < args.low)))
        for i, r in m[sel].iterrows():
            kw = None if kw_hit is None else bool(kw_hit[i])
            if big[i] and kw:
                reason, priority = "disagree + keyword", 1   # most likely a real miss in one table
            elif big[i]:
                reason, priority = "disagree", 2
            elif r[f"{f}_{na}"] < args.low and r[f"{f}_{nb}"] < args.low:
                reason, priority = "keyword, both low", 4    # often negation ("no effusion")
            else:
                reason, priority = "keyword, one low", 3
            flagged.append({ID: r[ID], "finding": f, "reason": reason, "priority": priority,
                            na: r[f"{f}_{na}"], nb: r[f"{f}_{nb}"], "diff": r[f"{f}_{na}"] - r[f"{f}_{nb}"],
                            "keyword_in_report": kw})

    summary = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.to_string(index=False))
    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out / "audit_summary.csv", index=False)
    cells = pd.DataFrame(flagged)
    if len(cells):
        cells = cells.sort_values(["priority", "finding"], kind="stable")
        print("\ncells by reason (review priority 1 first):")
        print(cells.groupby(["priority", "reason"]).size().to_string())
    cells.to_csv(args.out / "suspicious_cells.csv", index=False)
    print(f"\nwrote {args.out / 'audit_summary.csv'} and {args.out / 'suspicious_cells.csv'} "
          f"({len(cells)} cells; labels were not changed)")


if __name__ == "__main__":
    main()
