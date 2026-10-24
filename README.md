# Medical Report Label Refinement

An agent skill (Claude / Codex `SKILL.md` format) for improving **weak labels in medical imaging** when most studies only come with free-text radiology reports and only a few have expert ground truth.

It walks an agent through:

- **Extraction:** a descriptive schema (mention class × severity × supporting quote), with probabilities mapped in code and calibrated to expert positive rates
- **Multimodal image reads:** using frontier multimodal LLMs to read MRI slices for findings reports omit, and fusing them with text labels
- **Auditing:** multilingual keyword checks, source disagreement, repeated-read spread, calibration, ordinal and site-habit checks
- **Fusion and OOF correction:** cell-wise / z-score fusion and anchored out-of-fold self-distillation
- **Site bias:** a per-site logit offset used only during training to absorb reporting habits
- **Fair evaluation:** instrument noise, sealed folds, per-finding hidden-test probes, pre-registered gates, and what a score can and cannot show

For the image side (DICOM, laterality, ROI crops, study models, offline inference), see the companion skill [mri-study-pipeline](https://github.com/Longer430/mri-study-pipeline-skill).

## Origin

Distilled from the public discussions, released code and post-competition write-ups of the
[RSNA 2026 Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) Kaggle competition, plus a local paired comparison of two report-label sets. It credits the public work of stevenleehans, nartaa (blendjev labels), riadmohamed42 (JEV labels), Archit Konde, hengck23, Chris Deotte and Kirderf. The skill contains no competition data, reports, or labels.

## Example

See [examples.md](examples.md) for a full walk-through on the RSNA 2026 Knee Abnormality Detection competition: the public label lineage, an audit that found a systematic synovitis miss in an in-house label set, and a paired experiment design.

## Repository layout

```
SKILL.md                         the skill (what the agent reads)
examples.md                      RSNA 2026 Knee walk-through + how to reproduce it
scripts/audit_labels.py          compare two label tables + multilingual keyword screening
scripts/build_blend.py           cell-wise mean of two label sources (blendjev rule)
scripts/download_public_labels.py  fetch the public labels / train.csv from Kaggle (nothing is redistributed)
scripts/label_tools.py           calibrate / fuse / blend / decouple / probe / bootstrap / within-site
scripts/site_bias.py             training-only site bias (numpy init + PyTorch module)
scripts/selftest_label_tools.py  synthetic-data checks for the tools above
references/rsna-knee-writeups.md digest of the post-competition write-ups (own summaries, links)
toy_data/                        10 fictional studies to try the scripts in seconds
```

No competition data, reports or label files are stored here. Download them from Kaggle under their own licences.

## Install

**Claude Code / Claude:** put the folder containing `SKILL.md` at `~/.claude/skills/medical-report-label-refinement/`, or upload it as a skill.

**Codex:** put the folder at `~/.codex/skills/medical-report-label-refinement/`, restart the session, and invoke it with `$medical-report-label-refinement`.

The agent will then use it whenever you work on report-derived, LLM-extracted, or noisy labels for an imaging model. You can also ask for it by name.

## Scope and limits

- Guidance for research and ML competitions, not a clinical tool. Improved scores on a development reference are not clinical validation.
- The skill tells the agent not to overwrite original labels and not to send reports or images to external services without explicit approval.
- Heuristics such as "blend, don't replace" or "simple mean beats per-class picking" come from practitioners' reports on one competition. Validate them on your own data.

## License

Copyright (c) 2026 Wenting Long ([@Longer430](https://github.com/Longer430)).

Released under the [MIT License](LICENSE). You may use, modify and redistribute this skill, including commercially, as long as the copyright notice and license text are kept.

If this skill helps your work, a link back to this repository is appreciated.
