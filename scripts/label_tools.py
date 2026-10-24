#!/usr/bin/env python3
"""Label-refinement helpers distilled from the RSNA 2026 Knee write-ups.

Library functions (import them) plus a small CLI (one subcommand per tool).
All tables are wide CSV/parquet: one row per study, an ID column, one column per
finding. Nothing here edits an input file; every command writes a new file.

  calibrate   map report mention classes to shrunk expert positive rates (SKILL Section 1)
  fuse        z-score fusion of a text label with image/other reads, mapped back to the
              text distribution, plus an audit sidecar (SKILL Section 4)
  blend       anchored OOF self-distillation target: alpha*anchor + (1-alpha)*oof (Section 5)
  decouple    rank-space decoupling of coupled findings, e.g. the three OA columns (Section 10)
  probe-make  write column-inverted (or constant) copies of a submission (Section 11)
  probe-read  turn the leaderboard scores of those probes into per-finding AUCs
  bootstrap   paired, group-level bootstrap of the macro-AUC difference of two predictions
  within-site per-site AUC averaged over sites (for site-related changes)

Examples (run scripts/selftest_label_tools.py --out st first to get synthetic tables;
--findings is comma-separated and goes BEFORE the subcommand):
  python scripts/label_tools.py --findings "Medial OA,Lateral OA,PF OA,Effusion,Synovitis" calibrate --classes mention.csv --expert gold.csv --out calibrated.csv
  python scripts/label_tools.py fuse --text text.csv --other image_read.csv --out fused.csv
  python scripts/label_tools.py blend --anchor text.csv --oof oof.csv --alpha 0.3 --out round1.csv
  python scripts/label_tools.py bootstrap --truth gold.csv --a model_a.csv --b model_b.csv --groups groups.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ID = "StudyInstanceUID"
FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
            "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
MENTION_CLASSES = ["not_mentioned", "explicit_negative", "minor_or_borderline",
                   "present_below_rubric", "present_meets_rubric"]


# ----------------------------------------------------------------------------- io
def read_table(path: str | Path, id_col: str = ID) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if id_col not in df.columns:
        raise SystemExit(f"{path}: missing id column {id_col!r}")
    if df[id_col].duplicated().any():
        raise SystemExit(f"{path}: duplicate ids, e.g. {df.loc[df[id_col].duplicated(), id_col].iloc[0]}")
    return df.set_index(id_col)


def write_table(df: pd.DataFrame, path: str | Path, id_col: str = ID) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.reset_index().rename(columns={"index": id_col})
    if path.suffix == ".parquet":
        out.to_parquet(path, index=False)
    else:
        out.to_csv(path, index=False)


def findings_in(df: pd.DataFrame, findings=None) -> list[str]:
    cols = [f for f in (findings or FINDINGS) if f in df.columns]
    if not cols:
        raise SystemExit("no finding columns found; pass --findings")
    return cols


def safe_auc(y, p) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    ok = ~(np.isnan(y) | np.isnan(p))
    y, p = y[ok], p[ok]
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def macro_auc(truth: pd.DataFrame, pred: pd.DataFrame, findings) -> float:
    vals = [safe_auc(truth[f], pred[f]) for f in findings]
    return float(np.nanmean(vals))


# ----------------------------------------------------------------- 1. calibrate
def calibrate_mention_classes(classes: pd.DataFrame, expert: pd.DataFrame, findings,
                              prior_strength: float = 15.0, cap: float | None = None
                              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map each (finding, mention class) to a shrunk expert positive rate.

    classes: wide table of mention-class strings per study and finding.
    expert:  wide 0/1 table on the expert-labelled studies (subset of ids).
    level(t, g) = (k + m * p_g) / (n + m), where k of n expert studies in that
    finding/class are positive and p_g is the class's rate pooled over findings.
    Returns (calibrated wide table, mapping table with counts).
    """
    common = classes.index.intersection(expert.index)
    if len(common) == 0:
        raise SystemExit("no overlap between the class table and the expert table")
    long = []
    for f in findings:
        long.append(pd.DataFrame({"finding": f, "cls": classes.loc[common, f].values,
                                  "y": expert.loc[common, f].values}))
    long = pd.concat(long).dropna()
    pooled = long.groupby("cls")["y"].mean()
    rows = []
    for (f, g), grp in long.groupby(["finding", "cls"]):
        k, n = grp["y"].sum(), len(grp)
        level = (k + prior_strength * pooled[g]) / (n + prior_strength)
        rows.append({"finding": f, "cls": g, "n_expert": n, "k_pos": int(k),
                     "raw_rate": k / n, "pooled_rate": pooled[g], "level": level})
    mapping = pd.DataFrame(rows)
    if cap is not None:  # keep levels inside [1 - cap, cap], e.g. cap=0.95
        if not 0.5 < cap < 1:
            raise ValueError("cap must be in (0.5, 1)")
        mapping["level"] = mapping["level"].clip(1 - cap, cap)
    lut = {(r.finding, r.cls): r.level for r in mapping.itertuples()}
    out = pd.DataFrame(index=classes.index)
    for f in findings:
        out[f] = [lut.get((f, c), pooled.get(c, np.nan)) if pd.notna(c) else np.nan
                  for c in classes[f]]
    return out, mapping


# ---------------------------------------------------------------------- 2. fuse
def _zscore(x: pd.Series) -> pd.Series:
    sd = x.std(ddof=0)
    return (x - x.mean()) / sd if sd and np.isfinite(sd) else x * 0.0


def fuse_zscore(text: pd.DataFrame, others: list[pd.DataFrame], findings,
                weights: list[float] | None = None, map_back: bool = True
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per finding: average z-scores of the available sources, then (optionally)
    reassign values from the text label's own distribution by rank, so prevalence
    and pos_weight stay where the text put them. Missing sources are skipped, never 0.

    Returns (fused wide table, long audit sidecar with n_sources/spread/coverage).
    """
    sources = [text] + list(others)
    w = np.asarray(weights or [1.0] * len(sources), float)
    if len(w) != len(sources):
        raise SystemExit("--weights must have one value per source (text first)")
    ids = text.index
    fused = pd.DataFrame(index=ids)
    side = []
    for f in findings:
        Z = np.full((len(ids), len(sources)), np.nan)
        R = np.full((len(ids), len(sources)), np.nan)
        for j, s in enumerate(sources):
            if f not in s.columns:
                continue
            col = s[f].reindex(ids).astype(float)
            R[:, j] = col.values
            Z[:, j] = _zscore(col.dropna()).reindex(ids).values
        avail = ~np.isnan(Z)
        wz = np.where(avail, Z * w, 0.0).sum(1)
        wsum = np.where(avail, w, 0.0).sum(1)
        z = np.where(wsum > 0, wz / np.where(wsum > 0, wsum, 1), np.nan)
        if map_back:
            ref = np.sort(text[f].dropna().astype(float).values)
            ok = ~np.isnan(z)
            vals = np.full(len(ids), np.nan)
            if ok.any() and len(ref):
                q = (rankdata(z[ok]) - 0.5) / ok.sum()
                vals[ok] = np.quantile(ref, q)
            fused[f] = vals
        else:
            fused[f] = z
        side.append(pd.DataFrame({ID: ids, "finding": f,
                                  "n_sources": avail.sum(1),
                                  "coverage": wsum / w.sum(),
                                  "spread": np.nanmax(R, 1) - np.nanmin(R, 1)
                                  if np.isfinite(R).any() else np.nan,
                                  "fused": fused[f].values}))
    return fused, pd.concat(side, ignore_index=True)


# --------------------------------------------------------------------- 3. blend
def anchored_blend(anchor: pd.DataFrame, oof: pd.DataFrame, findings,
                   alpha: float | dict = 0.3) -> pd.DataFrame:
    """new = alpha * anchor + (1 - alpha) * oof. Always pass the ORIGINAL anchor,
    never the previous round's targets. Where OOF is missing, keep the anchor."""
    missing = anchor.index.difference(oof.index)
    if len(missing):
        print(f"warning: {len(missing)} anchor ids have no OOF prediction; keeping the anchor there")
    out = anchor[findings].copy().astype(float)
    o = oof.reindex(anchor.index)
    for f in findings:
        a = alpha[f] if isinstance(alpha, dict) else alpha
        blended = a * out[f] + (1 - a) * o[f]
        out[f] = blended.where(o[f].notna(), out[f])
    return out


# ------------------------------------------------------------------ 4. decouple
def pct_rank(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    ok = ~np.isnan(x)
    out[ok] = rankdata(x[ok]) / ok.sum()
    return out


def decouple(pred: pd.DataFrame, group: list[str], a: float = 0.2,
             pair_weights: dict | None = None) -> pd.DataFrame:
    """out_t = pct( pct(f_t) - a/(k-1) * sum_{o != t} w_to * pct(f_o) ) for t in group.
    With k = 3 this equals the 246th-place E-OA form (a/2 * sum). The output is a
    rank in (0, 1]; other columns are untouched. Choose `a` on an independent ruler
    for the sign and on a hidden test for the dose; start small (0.1-0.3)."""
    out = pred.copy()
    P = {t: pct_rank(pred[t].values) for t in group}
    k = len(group)
    for t in group:
        acc = np.zeros(len(pred))
        for o in group:
            if o == t:
                continue
            w = 1.0 if pair_weights is None else pair_weights.get((t, o), pair_weights.get((o, t), 1.0))
            acc += w * P[o]
        out[t] = pct_rank(P[t] - a / (k - 1) * acc)
    return out


# --------------------------------------------------------------------- 5. probes
def make_probes(sub: pd.DataFrame, findings, mode: str = "invert") -> dict[str, pd.DataFrame]:
    """One copy of the submission per finding with that column inverted (1 - p)
    or set to a constant (AUC 0.5). Submit each and record the displayed score."""
    probes = {}
    for f in findings:
        s = sub.copy()
        s[f] = 1.0 - s[f] if mode == "invert" else 0.5
        probes[f] = s
    return probes


def read_probes(base_score: float, probe_scores: dict[str, float], n_findings: int,
                mode: str = "invert", display_step: float = 0.001) -> pd.DataFrame:
    """invert:   AUC_f = 0.5 + (K/2) * (S_base - S_probe)
       constant: AUC_f = 0.5 + K * (S_base - S_probe)
    The +/- column is the worst-case error from display rounding of both scores."""
    k = n_findings
    factor = k / 2 if mode == "invert" else k
    rows = []
    for f, s in probe_scores.items():
        rows.append({"finding": f, "auc": 0.5 + factor * (base_score - s),
                     "plus_minus": factor * display_step})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- 6. bootstrap
def paired_bootstrap(truth: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame, findings,
                     groups: pd.Series | None = None, n_boot: int = 2000, seed: int = 0) -> dict:
    """Bootstrap the macro-AUC difference (b - a) on the same cases, resampling
    groups (patient / duplicate / site) rather than rows when groups are given."""
    ids = truth.index.intersection(a.index).intersection(b.index)
    t, pa, pb = truth.loc[ids, findings], a.loc[ids, findings], b.loc[ids, findings]
    g = (groups.reindex(ids) if groups is not None else pd.Series(ids, index=ids)).astype(str)
    uniq = g.unique()
    members = {u: np.where(g.values == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    base = macro_auc(t, pb, findings) - macro_auc(t, pa, findings)
    diffs = []
    tv, av, bv = t.values, pa.values, pb.values
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[u] for u in pick])
        da = np.nanmean([safe_auc(tv[idx, j], av[idx, j]) for j in range(len(findings))])
        db = np.nanmean([safe_auc(tv[idx, j], bv[idx, j]) for j in range(len(findings))])
        diffs.append(db - da)
    diffs = np.asarray(diffs)
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return {"n_cases": int(len(ids)), "n_groups": int(len(uniq)), "macro_a": macro_auc(t, pa, findings),
            "macro_b": macro_auc(t, pb, findings), "diff_b_minus_a": base,
            "ci95": [float(lo), float(hi)], "p_gain": float(np.nanmean(diffs > 0))}


# ---------------------------------------------------------------- 7. within-site
def within_site_auc(truth: pd.DataFrame, pred: pd.DataFrame, site: pd.Series, findings,
                    min_cases: int = 30) -> pd.DataFrame:
    """AUC computed inside each site, then averaged (sites with < min_cases or no
    positives are skipped). Use it when a change targets site/reporting habits:
    pooled AUC against report labels rewards the habit."""
    ids = truth.index.intersection(pred.index).intersection(site.index)
    s = site.reindex(ids)
    rows = []
    for f in findings:
        per = []
        for name, idx in s.groupby(s).groups.items():
            if len(idx) < min_cases:
                continue
            v = safe_auc(truth.loc[idx, f], pred.loc[idx, f])
            if np.isfinite(v):
                per.append(v)
        rows.append({"finding": f, "within_site_auc": float(np.mean(per)) if per else np.nan,
                     "n_sites": len(per), "pooled_auc": safe_auc(truth.loc[ids, f], pred.loc[ids, f])})
    df = pd.DataFrame(rows)
    return df


# ------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", default=ID)
    ap.add_argument("--findings", default=None, type=lambda v: [x.strip() for x in v.split(",") if x.strip()],
                    help="comma-separated finding columns (default: the 12 RSNA knee findings present)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate"); c.add_argument("--classes", required=True)
    c.add_argument("--expert", required=True); c.add_argument("--prior", type=float, default=15.0)
    c.add_argument("--out", required=True)

    f = sub.add_parser("fuse"); f.add_argument("--text", required=True)
    f.add_argument("--other", nargs="+", required=True); f.add_argument("--weights", nargs="*", type=float)
    f.add_argument("--no-map-back", action="store_true"); f.add_argument("--out", required=True)

    b = sub.add_parser("blend"); b.add_argument("--anchor", required=True)
    b.add_argument("--oof", required=True); b.add_argument("--alpha", type=float, default=0.3)
    b.add_argument("--out", required=True)

    d = sub.add_parser("decouple"); d.add_argument("--pred", required=True)
    d.add_argument("--group", nargs="+", default=["Medial OA", "Lateral OA", "PF OA"])
    d.add_argument("--a", type=float, default=0.2); d.add_argument("--out", required=True)

    pm = sub.add_parser("probe-make"); pm.add_argument("--sub", required=True)
    pm.add_argument("--mode", choices=["invert", "constant"], default="invert")
    pm.add_argument("--outdir", required=True)

    pr = sub.add_parser("probe-read"); pr.add_argument("--base", type=float, required=True)
    pr.add_argument("--scores", required=True, help='JSON {"Synovitis": 0.881, ...}')
    pr.add_argument("--mode", choices=["invert", "constant"], default="invert")
    pr.add_argument("--n-findings", type=int, default=12); pr.add_argument("--step", type=float, default=0.001)

    bs = sub.add_parser("bootstrap"); bs.add_argument("--truth", required=True)
    bs.add_argument("--a", required=True); bs.add_argument("--b", required=True)
    bs.add_argument("--groups", help="CSV with id and a 'group' column"); bs.add_argument("--n", type=int, default=2000)

    ws = sub.add_parser("within-site"); ws.add_argument("--truth", required=True)
    ws.add_argument("--pred", required=True); ws.add_argument("--site", required=True, help="CSV with id and 'site'")
    ws.add_argument("--min-cases", type=int, default=30)

    args = ap.parse_args()
    rd = lambda p: read_table(p, args.id)

    if args.cmd == "calibrate":
        cls = rd(args.classes); ex = rd(args.expert); fs = findings_in(cls, args.findings)
        out, mapping = calibrate_mention_classes(cls, ex, fs, args.prior)
        write_table(out, args.out, args.id)
        mapping.to_csv(Path(args.out).with_suffix(".mapping.csv"), index=False)
        print(mapping.round(3).to_string(index=False))
    elif args.cmd == "fuse":
        t = rd(args.text); others = [rd(p) for p in args.other]; fs = findings_in(t, args.findings)
        fused, side = fuse_zscore(t, others, fs, args.weights, not args.no_map_back)
        write_table(fused, args.out, args.id)
        side.to_csv(Path(args.out).with_suffix(".audit.csv"), index=False)
        print(f"wrote {args.out} and the audit sidecar; mean coverage {side.coverage.mean():.2f}")
    elif args.cmd == "blend":
        an = rd(args.anchor); fs = findings_in(an, args.findings)
        write_table(anchored_blend(an, rd(args.oof), fs, args.alpha), args.out, args.id)
        print(f"wrote {args.out} (alpha={args.alpha}, anchored to {args.anchor})")
    elif args.cmd == "decouple":
        p = rd(args.pred)
        write_table(decouple(p, args.group, args.a), args.out, args.id)
        print(f"wrote {args.out}; {args.group} are now ranks in (0, 1]")
    elif args.cmd == "probe-make":
        s = rd(args.sub); fs = findings_in(s, args.findings)
        for name, df in make_probes(s, fs, args.mode).items():
            write_table(df, Path(args.outdir) / f"probe_{args.mode}_{name.replace(' ', '_').replace(chr(39), '')}.csv", args.id)
        print(f"wrote {len(fs)} probe submissions to {args.outdir}")
    elif args.cmd == "probe-read":
        res = read_probes(args.base, json.loads(Path(args.scores).read_text()) if Path(args.scores).exists()
                          else json.loads(args.scores), args.n_findings, args.mode, args.step)
        print(res.round(4).to_string(index=False))
    elif args.cmd == "bootstrap":
        t = rd(args.truth); fs = findings_in(t, args.findings)
        g = rd(args.groups)["group"] if args.groups else None
        print(json.dumps(paired_bootstrap(t, rd(args.a), rd(args.b), fs, g, args.n), indent=2))
    elif args.cmd == "within-site":
        t = rd(args.truth); fs = findings_in(t, args.findings)
        res = within_site_auc(t, rd(args.pred), rd(args.site)["site"], fs, args.min_cases)
        print(res.round(4).to_string(index=False))
        print(f"macro within-site {res.within_site_auc.mean():.4f} | pooled {res.pooled_auc.mean():.4f}")


if __name__ == "__main__":
    main()
