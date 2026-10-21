# Worked example: RSNA 2026 Knee Abnormality Detection (Kaggle)

This walks through the skill on a real competition. All numbers are either quoted from public Kaggle posts and datasets (linked) or are aggregate counts from my own comparison. No reports, images, study IDs or label files are included here.

## The task

- Knee MRI studies; predict 12 findings per study (ACL, MCL, medial/lateral meniscus, medial/lateral/patellofemoral OA, effusion, synovitis, Baker's cyst, contusion, fracture). Metric: macro ROC-AUC.
- 4,407 training studies. **Only 58 have structured expert labels** ("gold-58"). The other 4,349 only have a free-text radiology report, in several languages (English, Spanish, ...).
- So the whole competition is a weak-label problem: turn reports into training targets, then train an image model on them.

## Step 0: Check the setup (Skill section 0)

| Item | What I found |
|---|---|
| Expert set | gold-58, read from the images. Reports can disagree with it: one public example has the report saying "no effusion" while gold says effusion = 1. |
| Is gold-58 still clean? | Many public pipelines use it for checkpoint or SWA selection, so it is a *development reference*, not independent validation. |
| Folds | The popular public folds file splits by study with no patient key. My own manifest groups by patient / duplicate imaging, so related studies never span folds. |
| Label versions | One in-house set ("v5", soft values plus a `__conf` weight column) and several public sets. Hashes recorded before any change. |

## Step 1: The public label lineage (Skill sections 1 and 3)

The strongest public labels were built by stacking independent report reads and averaging them:

```
stevenleehans v4_blend ─┐            (0.25 = "finding not addressed", ~19% of cells)
                        ├─ mean ─► labels_v1_blend ─┐
Gemini 3 Flash re-read ─┘  (Gemini only where v4=0.25)│
                                                     ├─ cell-wise mean ─► labels_v1_blendjev
JEV HYBRID (riadmohamed42) ──────────────────────────┘  (keep blend where JEV is NaN)
   = per-finding pick of stevenleehans / pilkwang labels
     + 0.05 × a 6-member LLM text ensemble
```

What the skill rules map to:

- **`not_addressed` is not negative:** v4_blend marks it with 0.25. The Gemini read fills it with a prior probability.
- **Status + probability + quote:** the Gemini re-read returned present / absent / not addressed, a calibrated probability and a report quote, prompted with the host's finding definitions.
- **Simple mean beats picking on a small set:** with the same CoAtNet recipe, gold-58 AUC was 0.888 with v4_blend alone, 0.904 with Gemini alone, and 0.904 with their mean. A per-finding pick chosen on gold did *not* beat the mean. The JEV author also found a per-target blend that scored higher on their eval set but dropped on the holdout, so they shipped a global ratio.
- In training, blendjev was used as plain soft BCE targets with weight 1 on every cell, with no prior fill and no per-cell weights (confirmed by the author).

Sources: [rsna-knee-hpo-assets](https://www.kaggle.com/datasets/nartaa/rsna-knee-hpo-assets), [jev-knee-labels](https://www.kaggle.com/datasets/riadmohamed42/jev-knee-labels), [rsna-knee-s6fall-weights training_code](https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights), [author Q&A](https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights/discussion/750180).

## Step 2: Audit my labels against blendjev (Skill section 2)

I compared in-house v5 with blendjev cell by cell on the 4,349 report-labelled studies.

**Overall the two are very close.** Pearson correlation was 0.91 to 0.99 across the 12 findings, and for most findings fewer than 1% of cells differed by more than 0.3.

**Synovitis was the exception:**

| Check | Result |
|---|---|
| Cells differing by > 0.3 | 204 |
| Direction | all 204: blendjev higher, v5 lower |
| Of those, report contains a synovitis keyword (`synovitis` / `sinovitis` / ...) | 193 |

So v5 was systematically missing explicitly reported synovitis, including in non-English reports (for example, a Spanish report containing "Sinovitis." scored 0.94 in blendjev and 0.2 in v5). A cheap keyword-plus-disagreement audit found it in minutes. Contusion showed 68 disagreements in the other direction (mostly v5 higher), which also deserves review.

A trap I hit: v5's rows for the gold-58 studies were the structured gold values copied in (AUC = 1.000 against gold). **They cannot be used to judge v5's label quality.** blendjev has no gold rows at all, so the two label sets cannot be compared on gold directly.

## Step 3: A paired experiment, not a label-vs-label score (Skill section 6)

Since labels can only be judged through the model they produce, I set up two arms that are identical except for supervision:

| | Baseline arm | New arm |
|---|---|---|
| Model | CoAtNet `coatnet_rmlp_2_rw_384`, ImageNet init, Attention-MIL over 12 three-slice windows | same |
| Cases / split | 3,862 grouped training candidates, fold 0 (3,138 train / 724 val) | same |
| Seed, epochs, LR, cache | 2026, 16 epochs, 2e-5 / 2e-4, FineSpacing80 cache | same |
| Training targets | v5 soft values | blendjev soft values |
| Loss weights | v5 `__conf` | 1 everywhere |
| Fold-0 metric reference | v5 (> 0.5) | v5 (> 0.5), so both arms use the same reference |
| Final judge | gold-58, k = 24 and 78 windows | same |

Because both values *and* weights change, this is reported as a **supervision-policy comparison**. To attribute any gain to the label values alone, a third arm (blendjev values + v5 weights) would be needed.

I smoke-tested first (8 cases, 1 epoch): the pipeline, label hash, output files and gold scoring all worked. The smoke AUC (~0.45) is meaningless and was ignored.

**Result:** pending. I will add the gold-58 per-class table here when the full run finishes.

## Step 4: OOF correction (Skill section 4), not done here

Public reports suggest the next gain is image-model OOF blending, especially for synovitis and lateral OA, which reports often leave out:

- Blending OOF predictions into labels helped; replacing labels with predictions was worse. A blend weighted heavily toward the model worked best for one competitor.
- In-fold predictions just echo the labels; they must be out-of-fold.
- Rounding labels to 0/1 hurt.

I skipped this because a full 5-fold OOF did not fit the time left in the competition.

## What the public scores do and do not show

- The 0.949 single model combined blendjev labels with external OAI data, crop and mirror TTA, and SWA epochs selected on gold-58. Label changes alone moved its public score from 0.940 to 0.942. Do not credit the full score to labels.
- Whether OAI was allowed external data was disputed in the forum (some participants could not register for access). Labels from blendjev do not involve OAI.
- Gold-58 has about 10 positives in some classes, so differences of a few hundredths are noise.

## Reproduce it yourself

This repository ships **no competition data and no label files**. You download them from Kaggle under their own licences.

**0. Try the scripts on fictional toy data first (no download, seconds):**

```bash
pip install -r requirements.txt
python scripts/audit_labels.py toy_data/toy_labels_a.csv toy_data/toy_labels_b.csv \
    --reports toy_data/toy_reports.csv --names a b --out audit_out
```

**1. Get the real files.** Join the competition and accept its rules on kaggle.com, set up a Kaggle API token, then:

```bash
python scripts/download_public_labels.py --out data
```

This fetches `labels_v1_blend.parquet` (CC0), `HYBRID_labels_scores.csv` (JEV, CC0), the published `labels_v1_blendjev.parquet` for checking, and the competition `train.csv` (competition rules apply; do not redistribute it).

**2. Rebuild blendjev and check it against the published file:**

```bash
python scripts/build_blend.py data/labels_v1_blend.parquet data/HYBRID_labels_scores.csv \
    audit_out/my_blendjev.csv --reference data/labels_v1_blendjev.parquet
# expect: rows 4349; the original author reports max |rebuild - reference| = 0 for this rule
```

**3. Audit your own label table against it** (same 12 columns plus `StudyInstanceUID`):

```bash
python scripts/audit_labels.py audit_out/my_blendjev.csv my_labels.csv \
    --reports data/train.csv --names blendjev mine --out audit_out
```

`audit_out/suspicious_cells.csv` lists cells to review, sorted by priority: 1 = the tables disagree and the report has a keyword (most likely a real miss), 2 = disagreement only, 3 = keyword but one table low, 4 = keyword but both low (mostly negations such as "no effusion"). With my in-house v5 table the script reproduced the Synovitis row exactly (204 disagreements, all "blendjev higher", 193 with a keyword) and put 219 cells in priority 1 across all findings, a review list one person can work through. Your numbers will depend on your labels.

Step 3 of the walk-through (the paired training run) depends on your own image pipeline, so it is described rather than scripted.

## Takeaways

1. Audit first: a keyword-plus-disagreement check found a systematic, fixable miss in minutes.
2. Average independent reads; don't tune per class on a tiny expert set.
3. Judge labels only through a paired model experiment on expert data or a hidden test.
4. Report exactly what changed (values, weights, mask) and what the result can and cannot prove.
