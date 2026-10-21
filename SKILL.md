---
name: medical-report-label-refinement
description: Improve weak labels for medical imaging models when most cases only have free-text radiology reports. Extract, audit, fuse, and OOF-correct soft labels, and evaluate label schemes fairly. Use when training image models on report-derived, LLM-extracted, or otherwise noisy labels.
license: MIT
---

# Medical Report Label Refinement

**When to use:** a medical imaging task where most studies only come with a free-text report (or with one or more public / LLM-extracted label sets), and only a small number of studies have expert ground truth. You need to turn reports into training targets, find label errors, fuse several label sources, and decide whether a new label set actually makes the image model better.

**Origin:** lessons from the RSNA 2026 Knee Abnormality Detection Kaggle competition. Credits go to the public work of stevenleehans, nartaa (blendjev), riadmohamed42 (JEV), Archit Konde, hengck23, Chris Deotte and Kirderf, plus a local paired comparison of two report-label sets. Report to the user in plain language.

## 0. Before starting (ask the user if anything is unclear)

- **Case manifest:** the role of every study (training candidate / expert-labelled holdout / review set / quarantine), plus a `group_id` (patient or duplicate imaging) and a fold. If there is no grouping, build it first so that one patient never spans two folds.
- **Expert ground truth:** how many cases, who labelled them (read the images or read the report?), and whether they were already used for tuning or model selection. If they were, call them a *development reference*, not independent validation.
- **Existing label versions:** path, hash, and the meaning of every column (value, confidence/weight, "not addressed" marker).
- Never overwrite original labels, raw data or earlier experiment outputs. Save each new version separately with its sources and hash.
- Never send reports, images or patient information to an external service (LLM APIs, embedding services included) without the user's explicit approval.

## 1. Extracting labels from reports: output format

For every (study, finding) cell, have the LLM or rule system return:

- **Status:** one of `present` / `absent` / `not_addressed`
- **Calibrated probability** in [0, 1]
- **Supporting quote** from the report (for auditing; guards against fabrication)

Put the task's official finding definitions in the prompt instead of letting the model use its own interpretation.

Key rules:
- **`not_addressed` is not negative.** Mark it explicitly (a fixed sentinel such as 0.25, or a separate mask column), or fill it with the finding's prior probability. Never fill it with 0.
- Keep soft probabilities. Hedged wording ("suspected", "cannot exclude") should map to intermediate values.
- Handle multilingual reports deliberately (Spanish, Portuguese, German, ...). Otherwise you get systematic misses.

## 2. Audit: find label errors (cheapest step, biggest payoff)

Do these in order. Each step produces a list of suspicious cells with report snippets for the user or a review process to confirm. Do not change labels automatically.

1. **Multilingual keyword check:** build a keyword list per finding (with negation handling). Flag cells where the report mentions the finding but the label is low, and cells where the label is high but the report says nothing.
2. **Disagreement between label sources:** count cells where two sources differ by more than 0.3, and check the direction. If one source is almost always higher, the other is probably missing that finding systematically.
3. **Disagreement across repeated reads:** have the LLM read the same report several times. Record `n_sources`, `spread` (max − min) and a confidence per cell, and re-read the source text for high-spread cells.
4. **Calibration check (on expert ground truth):** bin labels by value and check that the observed positive rate in each bin is close to the bin's value. Labels around 0.3 should be positive about 30% of the time.
5. **Expect a built-in gap between reports and ground truth.** If ground truth was read from the images, the report may simply not mention some findings (synovitis and diffuse disease are typical). That is not an extraction error. Report-only labels have a ceiling here, and only image-model OOF correction (Section 4) can get past it.

Audit example (pandas; adjust column names):

```python
import re, pandas as pd

kw = {"Synovitis": r"s[iy]novi|synovial thicken"}  # add multilingual terms for each finding
m = (labels_a.merge(labels_b, on="StudyInstanceUID", suffixes=("_a", "_b"))
             .merge(reports, on="StudyInstanceUID"))
for f, pat in kw.items():
    hit = m.Report.fillna("").str.contains(pat, flags=re.I)
    diff = m[f + "_a"] - m[f + "_b"]
    print(f, "disagree>0.3:", int((diff.abs() > .3).sum()),
          "a higher:", int((diff > .3).sum()),
          "keyword present but b<0.5:", int((hit & (m[f + "_b"] < .5)).sum()))
```

## 3. Fusing several label sources

- **Default: cell-wise simple mean** of independent soft-label sources. Where one source is missing (NaN or its "not addressed" marker), use the others.
- Do not pick "the best source" per finding or tune per-class weights on a small expert set. With roughly 10 positives per class, that fits noise and drops on a holdout. If you must weight, use a single global ratio.
- Switching to a different LLM usually gains little. The gain comes from fusing several independent reads.

```python
import numpy as np

b = blend[LAB].to_numpy(float)
j = other.reindex(blend.StudyInstanceUID)[LAB].to_numpy(float)
out = np.where(np.isnan(j), b, (b + j) / 2)
```

## 4. Correcting labels with image-model OOF predictions (getting past the report ceiling)

- Use **out-of-fold** predictions only, i.e. from a model whose training folds did not include that case. In-fold predictions just echo the original labels.
- **Blend, do not replace:** `new = alpha * label + (1 - alpha) * oof_pred`. Replacing is usually worse. The useful alpha can lean heavily toward the model (50/50 already helps in practice and leaning further can add a little), but validate it.
- The largest gains are on findings that reports often leave out (synovitis, diffuse changes). You can apply this to those classes only.
- Masking low-confidence cells alone usually does nothing. The model needs more positive signal.
- This is pseudo-labelling. Say so when reporting a "single model" result.
- It needs full K-fold OOF, which costs about K training runs. Check the time budget first.

## 5. Using labels in training

- Use soft labels directly as BCE targets. **Do not binarise them** (rounding to 0/1 measurably hurt).
- Make three things explicit and record them: target values, per-cell loss weights, and the mask. Treat `not_addressed` with a prior value or a mask, never as 0.
- Compute `pos_weight` from the supervision labels on the training side.
- Optionally initialise the output bias to the logit of the mean training label (speeds up early convergence).

## 6. Evaluating label schemes (the most important discipline)

- **Do not compare label schemes against report-derived labels.** A model always scores best on the labels it was trained on, so that comparison is biased.
- The only usable judges are expert ground truth (small and noisy, so treat it as a development reference) and an external hidden test such as a public leaderboard. Treat differences within a few hundredths on a small expert set as noise. Use bootstrap CIs or several seeds when possible.
- **Paired experiments:** fix the cases, split, cache/preprocessing, initialisation, seed, training budget and evaluation reference. Change only the supervision.
- If values, weights and mask all change at once, call it a *supervision-policy comparison*. To attribute the effect, add an arm such as "new label values + old weights".
- Label AUC (labels vs. a reference) and image-model AUC measure different things. Do not compare their magnitudes, and do not credit all of a model's gain to the labels.
- Do not pick checkpoints or SWA epochs on the expert set and then report scores on the same expert set. Select inside an inner fold or use a fixed epoch.
- Smoke-test first (a few cases, 1 epoch) to confirm the pipeline, logs, output files and label hash, then run the full job. Smoke-test AUC means nothing.

## 7. Recommended workflow (ordered by return on effort)

1. Check the manifest, grouping and how the expert set has been used. Record the current label hashes.
2. Run the multilingual keyword audit and the source-disagreement audit, then produce a list of suspicious cells, get them confirmed by the user or reviewer, and save a new label version.
3. Average all trusted sources cell by cell to get a candidate label set.
4. Run a paired experiment, old labels vs. candidate labels, on a single fold. Report per-class and macro AUC on the expert set.
5. If time allows, add OOF soft blending, starting with findings that reports tend to omit.
6. When reporting, separate "code implemented / smoke run passed / full run complete / effect validated", and list per-class metrics, sample sizes and limitations.

## 8. Report template (plain language)

- **What changed:** label sources, fusion rule, weight/mask policy
- **How it was compared:** paired setup, and exactly which factors changed
- **Results:** expert-set macro AUC and per-class AUC with positive/negative counts; public leaderboard if available
- **What it does and does not show:** sample size, whether the expert set was used for selection, whether this is independent clinical validation
- **Suggested next step**


For a full worked example, see `examples.md` in this skill's folder.

---

Copyright (c) 2026 Wenting Long ([@Longer430](https://github.com/Longer430)). Released under the MIT License.
