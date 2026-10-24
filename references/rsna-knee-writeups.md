# RSNA 2026 Knee: what top write-ups did on the label side

Digest of the post-competition write-ups on the [discussion forum](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion), first two pages, read 2026-10-23. Summaries are in my own words; numbers are as the authors reported them (not reproduced here). Follow the links for the full posts. Image-side details (inputs, ROI crops, models, inference) are in the `mri-study-pipeline` skill's own digest.

Setting: 4,407 training studies, 58 with expert image labels ("gold58"), 9 report languages, hidden test of about 1,300 studies (30% public / 70% private) graded by MSK radiologists reading images, macro AUC over 12 findings. Private #1 0.960, gold line about 0.955; almost everyone lost 0.009 to 0.013 from public to private.

## Per-team digest

### 21st (solo gold) — [writeup](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/21st-place-solo-gold-solution-llm-soft-labels)

- Built an MCP server ("RSNA Knee Reader") so a multimodal LLM in a chat client could page through each study's series on the server: four-slice preview pages, native-resolution single slices on demand, the report as context, 12 probabilities returned as JSON. One main agent plus three subagents in parallel across studies. Tool released on GitHub (zhuzhirui2005/rsna-knee-reader).
- Key prompt change: removed "probability 1 when the report confirms it"; images first, report only as context, each target independent, `null` allowed.
- Labels on gold58: 4-source report consensus 0.896 macro AUC, LLM image-first soft labels 0.935 (but lower F1 at 0.5, so they kept probabilities).
- Matched training comparison, same splits and epochs: ConvNeXt-Small 0.915 → 0.943 on gold58 when only the labels changed.
- Self-distillation target 0.75 × LLM label + 0.25 × in-sample teacher; continuing two epochs on it beat continuing on LLM labels by +0.0005 OOF (paired bootstrap CI above zero); full retrain from ImageNet on it scored best public (0.963).
- Per-target blend weights tuned on OAI or gold+OAI lost on submission; final used fixed weights.

### 22nd (solo gold, private 0.954, public 0.965) — [writeup](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/22nd-place-solo-gold-solution)

- Report reading by one LLM with explicit "positive if / negative if" per finding and a default for unmentioned findings; told not to trust grade strings literally. gold58 0.892.
- Image reading by another LLM on per-series montages (tiles labelled only by series and slice; metadata and report given as text, report marked as possibly wrong). gold58 0.929. Used for 10 of 12 findings; lateral OA kept the report read (rubric needs >50% thickness over >1 cm, hard to see).
- Synovitis got its own procedure: two landmark crops (Hoffa fat pad, suprapatellar pouch), six graded signs, ranked and quantile-mapped onto an earlier model's OOF distribution. gold58 0.869.
- Training target = 0.5 × LLM read + 0.5 × 5-fold OOF; loss weight 0.25 + 0.75 × LLM confidence; gold58 rows keep true labels. Three label sets scored 0.954 → 0.962 → 0.967 on gold58 (10 findings).
- Never selected on gold58: every decision on 4,342-study OOF scored against the image read, with written thresholds (e.g. mean gain 0.0015 with bootstrap CI above 0; seed noise ~0.003). Single-model gain from swapping labels (+0.0031) matched the public gain (+0.003).
- Left out sex, scanner, coil, field strength as inputs: sex helped the proxy metric and hurt gold58.
- Things that did not move the ensemble: re-labelling lateral OA/meniscus from compartment crops (labels improved on gold58, models flat), a fluid-volume effusion feature (passed once, then redundant after better labels), decoupling synovitis from effusion (the apparent coupling was a binary-vs-continuous correlation artefact).

### 32nd (private 0.952, public 0.964) — [post](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/751930)

- Teacher = 0.5 × mean of three LLM report reads + 0.5 × VLM image read (normal and zoomed views averaged); gold58 rows use expert labels.
- For 847 site × finding cells with strong reporting habits (e.g. synovitis and effusion, lateral and PF OA over-reported), used the image read at 100%.
- Single model on report-only labels 0.943 public; with image read plus an evidence loss (attention pulled toward slices the VLM cited) 0.960.
- Lessons: the teacher matters most; a pseudo-label metric can be biased toward your own models; no new code on the last day.

### 33rd (private 0.952, public 0.966) — [writeup](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/33rd-place-silver-solution-no-external-data)

- Report labels: LLM structured extraction (severity, not mentioned vs explicit negative, acute/chronic), mapped to positive rates among gold58 with shrinkage, averaged with a public label set.
- "Not mentioned" cells were almost constant (75% of synovitis targets), so only there they spread targets with a teacher: y' = sigmoid(logit(y) + 0.5 × z), z = teacher logit standardised per finding.
- Training-only "trap": logit = image_logit + a[site, finding] + mean over series of g(acquisition params); inference uses image_logit only. Site = report language × scanner, 44 pseudo-sites; a has a N(0, 0.5²) penalty and empirical-Bayes init; g is a zero-init MLP.
- Ablation (one change per row, public/private): no trap 0.949/0.932 → + site term 0.955/0.940 → + acquisition branch 0.960/0.946 → + spread "not mentioned" 0.962/0.947.
- Turning off acquisition-parameter inputs at inference alone moved an older model 0.947 → 0.959 public. Acquisition parameters predict report language with 98% accuracy.

### 38th (private 0.952, public 0.964) — [summary](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/751924), [full](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/the-hospital-is-a-variable-too)

- Report groups vs expert-positive rate on gold58: not mentioned 9/144 = 0.06, negative 9/197 = 0.05, mild/borderline 32/96 = 0.33, positive below rubric 48/68 = 0.71, positive meets rubric 128/150 = 0.85.
- Steps that moved the public LB: "not mentioned" → 0.05 at half weight for 11 findings (synovitis excluded) +0.012; mild → ~0.33 ordered by the image teacher +0.007 (+0.003 more for severity 1–2 positives); half report, half image vote on addressed findings +0.007; shrink confident groups toward the expert rate (prior m = 15) +0.002 public / +0.003 private. Moving groups away from the expert rate lost every time.
- Training-only hospital offset B[site, finding], lr 5e-3, no weight decay; 9 language groups, then 29 sites (language × scanner × field strength × series set). 0.949/0.935 → 0.953/0.938, 29 sites → 0.955/0.941.
- Reverse check: adding the learned offsets back at test time via a header-based site classifier (98% accurate) dropped the same checkpoint to 0.950/0.936.
- Diagnostic: eta-squared of validation logits grouped by site; its change correlated −0.54 with LB change over 46 steps.
- Report-based validation misjudged hospital-related changes; only the expert-graded LB could judge them.

### 49th (solo, private 0.951, public 0.962) — [writeup](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/49th-place-silver-solo-whole-study-2-5d-reader), [code](https://github.com/Komil-parmar/rsna-knee-2026-49th-place)

- Descriptive extraction (mention class × severity, probability set in code: explicit normal 0.03, not mentioned 0.15, minor 0.40, present 0.62/0.80/0.95 by severity) beat rubric verdicts (0.839 vs 0.809).
- Four readers averaged (two local Qwen models, a public set, a hosted Claude Sonnet pass) 0.899; a distilled XLM-R/LaBSE classifier 0.885. Report-only ceiling ≈ 0.90.
- Same prompt and model at temperature 0 gave 0.810 and 0.872 in two batch configurations.
- Label rounds y = 0.3 × text + 0.7 × 5-fold OOF, always re-anchored to the same text: v10 +0.006, v11 +0.003, v12 flat, v13D/v14 (with OA decoupling in the teacher) +0.003/+0.001. Single-fold teachers tied twice.
- Sealed fold audit: clean label-round gain +0.004 report-OOF / +0.016 gold58; leaky setup showed +0.011, about 3× too high.
- Measured noise: gold58 paired macro ±0.0125; fold-0 seed 0.0026. Column-inversion probe: AUC_f = 0.5 + 6 × (S_base − S_probe).
- Rules after the blend: synovitis composite (+1 tick), scanner-habit removal, OA decoupling (LatOA − 0.15 × MedOA in rank space).
- Negative results table: confident-learning mask, PU loss for synovitis, per-finding label variants, gold58-fitted blends (−0.001 to −0.009), bag-level distillation (its validation gain was leakage).

### 69th (Civitasmass, private 0.943 own single model) — [writeup](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/multimodal-llm-relabeling-oof-self-distillation)

- gold58: report-only 0.90, montage-only image read 0.877, 50/50 average 0.922, report and montage in one prompt 0.932; mixing OOF back in 0.945.
- Images helped most on effusion, contusion, Baker's, medial OA; reports were stronger on ACL, MCL, menisci, PF OA, synovitis.
- Expert cases as hard targets: neutral to slightly negative; as few-shot examples: 0.947 vs 0.951; fitting fusion weights on them: overfit.
- Seed redraw moved fold CV by 0.0025; 3 of 32 controlled experiments cleared a pre-committed gate.
- Label diversity added ~+0.001 public per step; a second seed of the same model and labels added nothing.
- Agentic multi-step LLM readers did not beat a plain multimodal prompt; mapping size words to the rubric hurt.

### 72nd (private 0.949, public 0.960) — [writeup](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/72nd-place-training-only-site-bias-and-image-read)

- Site bias: site = report language (langid) × manufacturer, 24 groups (<30 studies share one bias); loss BCE(y, sigmoid(f(x) + b[site])); b init logit(group mean) − logit(global mean), lr 1e-2, no weight decay; dropped at inference. +0.007 private, replicated with another seed and at 384 px; bias on 7 subjective findings only gave +0.004.
- Retraining the label teachers with the bias: +0.002/+0.003 more.
- One label teacher per public report-label table, then average their OOF: 0.923 → 0.931 private at the same model and epochs.
- Pooled OOF AUC vs report labels fell with the bias (0.912 → 0.906) while within-site AUC stayed flat and the LB rose; switched to within-site AUC.
- Last-day image-reading LLM labels (three 12-slice montages + report + definitions, one call per study, ~21k tokens, 45 minutes at 128 parallel calls): gold58 0.928 vs 0.903 report-only with the same prompt; private 0.943 → 0.949.
- More text readers did not help: four LLMs scored 0.86–0.89 and correlated 0.91–0.98 with each other.

### 74th (private 0.949, public 0.962) — [writeup](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/74th-place-knee-centred-views-llm-labels-and-oa)

- Keeping "not mentioned" apart from "explicitly normal": 0.863 → 0.904 on gold58.
- Each report statement mapped to a level; levels checked against OAI-trained image models and out-of-order levels merged into neighbours.
- Pseudo-label rounds mix 0.4 label + 0.6 OOF by rank only (values keep the report's prevalence).
- LLM image readers accepted per finding only after passing gold58 rules (e.g. a 50/50 mix no worse than the label). The same reader lost 0.033 macro AUC when it could not crop and zoom. Each accepted group ≈ +0.002 LB.
- OAI-trained readers (MOAKS mapped to the findings) used where they reached 0.85 on gold58; they over-coupled medial meniscus and medial OA, so used only for effusion, synovitis, Baker's.
- Continuing existing models 2–3 epochs at 0.3× LR on new labels: gold58 0.950 vs 0.941–0.944 from scratch; label change +0.0096 (95% CI +0.0016 to +0.0175) vs extra epochs +0.0023.
- Selecting epochs or weights on gold58 gave +0.003 there and −0.001 on the LB; afterwards used it only as a veto. Rule-based relabelling from report wording: −0.020 LB.

### 76th (private 0.948, public 0.963) — [writeup](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/rsna-knee-ai-challenge-solution)

- Two local open LLMs (Qwen3-14B once, MedGemma-27B five times) averaged: gold58 0.8945. Prompt framing "labels come from experts re-reading the images; reports over-call effusion and miss synovitis" was the only rewrite that helped.
- Two independent extractors agreed on 88% of their disagreements with the experts; 71% of gold synovitis positives had no synovitis term.
- Five teacher–student generations with y = 0.5 × report + 0.5 × OOF of gold-fine-tuned teachers; gold fine-tune = frozen trunk, slice transformer and head for 2 epochs at 5e-5.
- Trunk interpolation 0.7 × trained + 0.3 × RadImageNet init: +0.002 private on both a 5-model family and a 40-model ensemble.
- A re-extracted label set with +0.022 on gold58 lost on both leaderboards. Ensembles of nine 0.963 entries scored below their best single family on private.

### 102nd (private 0.947) — [writeup](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/102nd-place-local-llm-report-labeling-and-anatomy)

- Only local LLMs (MedGemma-27B text, Gemma-4-26B), five responses on the original report and five on its English translation per model; five states mapped to 1 / 0.75 / 0.25 / 0 / 0.
- Weak-label heuristics on "not mentioned" synovitis (copy explicit effusion; probable positive if any OA explicit). Synovitis was still the weakest finding on private.
- Per-finding LB AUC estimated after the deadline by setting one column to a constant.

### 220th (private 0.944) — [writeup](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/220th-place-silver-gold-batch-fine-tuning-on-to)

- Drew gold58 studies into batches with probability p instead of up-weighting their loss (with batch size 2 most batches had none); final p = 0.5. Weight soup of the public checkpoint (0.25) and four gold-batch fine-tunes (0.1875 each).
- Loss weight 0.3 on report-silent cells of findings radiologists often leave unmentioned.

### 228th (private 0.944) — [writeup](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/228-solo-silver-subtract-the-report-style-prior)

- Fitted a metadata-only model (logistic regression + shallow gradient boosting on DICOM header fields) to predict the report labels; 5-fold AUC 0.63–0.77.
- Adding the prior to the blend lost (0.958 → 0.954 public); subtracting a small share of its rank from the blend gained. Confirms the habit signal is anti-informative on expert labels.

### 246th (private 0.944) — [writeup](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/246th-place-silver-report-labels-are-not-expert)

- OA compartment coupling (Spearman/tetrachoric, M-L / M-PF / L-PF): blend scores .85/.82/.81, report labels .83/.80/.77, gold58 .51/.56/.45, OAI MOAKS .12/.13/−.01.
- E-OA: out_t = pct(pct(f_t) − (a/2) Σ w_t,o pct(f_o)), symmetric or pair-weighted by excess coupling; a ≈ 0.2–0.3 best on the LB. Checked first on 2,399 OAI knees with a negative-a control (all folds positive).
- gold58 and raw OAI overstated the public gain about 2.4–3×; the coupling-matched estimate understated it. Independent rulers gave the sign, the LB gave the dose.
- Engineering: one pre-registration file per entry, byte-identical all-off controls, canary columns.

### Discussion threads

- **Astra Is All You Need? (tingyi)** — [post](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/751920). Five models blind-read gold54 per finding group. Blind reads fused better with the text label than report-aware reads (effusion 0.962 vs 0.926–0.931). One study per call (synovitis 0.756 → 0.800), one group per call (all-in-one cost 4–5× and lost 0.08 on lateral/PF OA), reasoning effort xhigh best, two reads averaged, z-score fusion with rank map-back to the text distribution. Fracture: zoomed local sheets with boxes on the 748 text-uncertain studies, label 0.906 → 0.928. Model on these labels: gold58 0.916 → 0.935, public 0.946 → 0.950. Note: these per-model results reflect that author's setup and model versions; re-test before relying on them.
- **Chris Deotte (243rd)** — [writeup](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/243rd-silver-medal). Report + mosaic in one request per study for three frontier models; best two-model blend 0.9275 on gold58; image-only reads with that setup 0.74–0.75. Labels published as a Kaggle dataset (cdeotte/rsna-knee-frontier-llm-hybrid-labels).
- **LLM label-extraction benchmark (Kirderf)** — [thread](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747658). Benchmarks of report and vision prompts; k256.dev: Opus in Claude Code subagents reading images + report 0.927, with a per-finding tool-building harness 0.932 at much higher token use; hengck23 pointed out a gold study whose report says no effusion while the label is effusion = 1.
- **Best single-model score** — [thread](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304). Mid-competition consensus that labels were the lever; Archit Konde: mix OOF into labels rather than replace, keep soft labels, check calibration on gold58; Chris Deotte: a "single model" on pseudo-labels is not comparable to one on report labels.
- **How do top Kagglers hold up on private?** — [thread](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/751979). Dieter: no fixed recipe; pick one best-LB and one best-CV submission; all diversity helps; re-validate anything from public notebooks in your own CV.

## Numbers worth remembering

| Quantity | Value | Source |
| --- | --- | --- |
| Report-only label ceiling vs image-reading experts | ≈ 0.90 macro AUC | 49th, 76th, 412th, many |
| Best image-aware label sets on gold58 | 0.928–0.967 | 72nd, 22nd |
| gold58 paired macro noise | ≈ ±0.012 | 49th, Civitasmass |
| Single-fold seed noise | ≈ 0.0025 | 49th, Civitasmass |
| Training-only site bias | +0.006 to +0.007 private | 72nd, 38th, 33rd |
| Public → private drop | 0.009–0.013 for most teams; ~0.016 for public-notebook copies | 49th, Northstar |
| Share of a label gain that reaches the model | 13–18% | 38th |
