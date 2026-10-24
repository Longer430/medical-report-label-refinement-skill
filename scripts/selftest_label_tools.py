#!/usr/bin/env python3
"""Self-test for label_tools.py and site_bias.py on synthetic data (no real data).

Builds a fake cohort with sites whose reports over- or under-call findings, an
expert-labelled subset, report mention classes, a text label, an image read and
two models, then checks that every tool behaves as documented. Also writes the
synthetic tables to --out so the CLI can be tried on them.

  python scripts/selftest_label_tools.py --out selftest_out
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_tools as lt  # noqa: E402
import site_bias as sb  # noqa: E402

F = ["Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis"]


def synth(n=1500, n_sites=6, n_gold=300, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"SYN-{i:05d}" for i in range(n)]
    site = rng.integers(0, n_sites, n)
    habit = rng.normal(0, 1.0, (n_sites, len(F)))           # reporting habit per site and finding
    latent = rng.normal(0, 1, (n, len(F)))
    latent[:, 1] += 0.8 * latent[:, 0]                       # lateral OA correlates with medial OA
    latent[:, 2] += 0.8 * latent[:, 0]
    truth = (latent > 0.7).astype(float)                     # what experts would see on images
    report_score = latent + habit[site] + rng.normal(0, 0.8, latent.shape)
    cls = np.select([report_score < -0.5, report_score < 0.3, report_score < 0.9, report_score < 1.5],
                    ["not_mentioned", "explicit_negative", "minor_or_borderline", "present_below_rubric"],
                    "present_meets_rubric")
    text = 1 / (1 + np.exp(-(report_score - 0.7)))
    image = 1 / (1 + np.exp(-(latent - 0.7 + rng.normal(0, 0.9, latent.shape))))
    image[rng.random(image.shape) < 0.2] = np.nan            # image read missing for some cells
    model_a = latent + rng.normal(0, 1.2, latent.shape)
    model_b = latent + rng.normal(0, 0.8, latent.shape)      # b is genuinely better
    habit_model = latent + habit[site] * 1.5 + rng.normal(0, 0.8, latent.shape)  # learned site shortcut
    mk = lambda a: pd.DataFrame(a, index=pd.Index(ids, name=lt.ID), columns=F)
    gold_ids = ids[:n_gold]
    return dict(ids=ids, site=pd.Series(site, index=ids, name="site"), truth=mk(truth),
                gold=mk(truth).loc[gold_ids], classes=mk(cls), text=mk(text), image=mk(image),
                a=mk(model_a), b=mk(model_b), habit=mk(habit_model), habit_true=habit)


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        check.failed += 1


check.failed = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="selftest_out")
    out = Path(ap.parse_args().out)
    d = synth()

    # 1. calibrate: levels must be close to the true positive rate of each class
    cal, mapping = lt.calibrate_mention_classes(d["classes"], d["gold"], F, prior_strength=15)
    true_rate = []
    for f in F:
        for g, grp in d["truth"][f].groupby(d["classes"][f]):
            true_rate.append({"finding": f, "cls": g, "true": grp.mean()})
    m = mapping.merge(pd.DataFrame(true_rate), on=["finding", "cls"])
    err = (m.level - m.true).abs().mean()
    check("calibrate: levels track true class positive rates", err < 0.08, f"mean abs err {err:.3f}")
    check("calibrate: classes ordered (not mentioned < meets rubric)",
          (mapping.groupby("cls")["level"].mean()["not_mentioned"] <
           mapping.groupby("cls")["level"].mean()["present_meets_rubric"]))

    # 2. fuse: map-back keeps the text distribution, missing image cells are not zeros
    fused, side = lt.fuse_zscore(d["text"], [d["image"]], F)
    for f in F[:1]:
        q_text = np.quantile(d["text"][f], [0.1, 0.5, 0.9])
        q_fused = np.quantile(fused[f], [0.1, 0.5, 0.9])
        check("fuse: map-back preserves the text distribution", np.allclose(q_text, q_fused, atol=0.02))
    auc_text = lt.macro_auc(d["truth"], d["text"], F)
    auc_fused = lt.macro_auc(d["truth"], fused, F)
    check("fuse: adding an independent image read raises label AUC", auc_fused > auc_text,
          f"{auc_text:.3f} -> {auc_fused:.3f}")
    check("fuse: coverage reflects missing sources", 0.85 < side.coverage.mean() < 0.95,
          f"mean coverage {side.coverage.mean():.2f}")

    # 3. blend: formula and fallback
    oof = d["b"].apply(lambda c: 1 / (1 + np.exp(-c)))
    oof_missing = oof.drop(index=d["ids"][:10])
    bl = lt.anchored_blend(d["text"], oof_missing, F, alpha=0.3)
    i = d["ids"][20]
    check("blend: 0.3*anchor + 0.7*oof", np.isclose(bl.loc[i, F[0]], 0.3 * d["text"].loc[i, F[0]] + 0.7 * oof.loc[i, F[0]]))
    check("blend: missing OOF keeps the anchor", np.isclose(bl.loc[d["ids"][0], F[0]], d["text"].loc[d["ids"][0], F[0]]))

    # 4. decouple: lower OA co-firing, ranks in (0, 1]
    oa = ["Medial OA", "Lateral OA", "PF OA"]
    dec = lt.decouple(d["habit"], oa, a=0.3)
    c0 = d["habit"][oa].corr(method="spearman").values[np.triu_indices(3, 1)].mean()
    c1 = dec[oa].corr(method="spearman").values[np.triu_indices(3, 1)].mean()
    check("decouple: reduces OA rank correlation", c1 < c0, f"{c0:.2f} -> {c1:.2f}")
    check("decouple: outputs are ranks in (0, 1]", dec[oa].min().min() > 0 and dec[oa].max().max() <= 1)

    # 5. probes: recover per-finding AUC from (rounded) macro scores
    sub = d["b"]
    score = lambda s: round(lt.macro_auc(d["truth"], s, F), 3)
    base = score(sub)
    probes = lt.make_probes(sub, F, "invert")
    res = lt.read_probes(base, {f: score(p) for f, p in probes.items()}, len(F), "invert")
    true_auc = {f: lt.safe_auc(d["truth"][f], sub[f]) for f in F}
    worst = max(abs(r.auc - true_auc[r.finding]) for r in res.itertuples())
    check("probe: inverted-column read-out within rounding error", worst <= res.plus_minus.max() + 1e-9,
          f"worst error {worst:.4f}, bound {res.plus_minus.max():.4f}")

    # 6. bootstrap: b beats a with a CI above zero
    groups = pd.Series([f"g{i // 2}" for i in range(len(d["ids"]))], index=d["ids"])  # pairs share a group
    bs = lt.paired_bootstrap(d["truth"].loc[d["gold"].index], d["a"], d["b"], F, groups, n_boot=300)
    check("bootstrap: better model has positive diff and CI above 0", bs["ci95"][0] > 0,
          f"diff {bs['diff_b_minus_a']:.3f}, CI {bs['ci95'][0]:.3f}..{bs['ci95'][1]:.3f}")

    # 7. within-site: a site-shortcut model looks better pooled against report labels
    report_truth = (d["text"] > 0.5).astype(float)
    ws = lt.within_site_auc(report_truth, d["habit"], d["site"], F, min_cases=30)
    gap = (ws.pooled_auc - ws.within_site_auc).mean()
    check("within-site: shortcut inflates pooled AUC vs report labels", gap > 0, f"pooled - within = {gap:.3f}")

    # 8. site bias init
    keys = sb.build_site_keys(pd.DataFrame({"lang": d["site"].astype(str), "vendor": "x"}), ["lang", "vendor"], 30)
    codes = pd.factorize(keys)[0]
    init = sb.init_site_bias(d["text"].values, codes)
    corr = np.corrcoef(init[:, 0], [d["habit_true"][int(k.split("|")[0]), 0] for k in pd.factorize(keys)[1]])[0, 1]
    check("site bias: init offsets track the true site habit", corr > 0.8, f"corr {corr:.2f}")
    if sb.SiteBias is not None:
        import torch
        m = sb.SiteBias(init.shape[0], init.shape[1], init)
        x = torch.zeros(4, init.shape[1]); s = torch.tensor([0, 1, 2, 3])
        m.train(); tr = m(x, s); m.eval(); ev = m(x, s)
        check("site bias: applied in train, dropped in eval", tr.abs().sum() > 0 and ev.abs().sum() == 0)
    else:
        print("SKIP site bias torch module (torch not installed)")

    # write synthetic tables for trying the CLI
    out.mkdir(parents=True, exist_ok=True)
    for k in ["classes", "gold", "text", "image", "a", "b", "habit"]:
        lt.write_table(d[k], out / f"{k}.csv")
    pd.DataFrame({lt.ID: d["ids"], "site": d["site"].values}).to_csv(out / "site.csv", index=False)
    pd.DataFrame({lt.ID: groups.index, "group": groups.values}).to_csv(out / "groups.csv", index=False)
    print(f"\nsynthetic tables written to {out}/")
    print("FAILED" if check.failed else "ALL CHECKS PASSED", f"({check.failed} failures)")
    sys.exit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
