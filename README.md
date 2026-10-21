# Medical Report Label Refinement

An agent skill (Claude / Codex `SKILL.md` format) for improving **weak labels in medical imaging** when most studies only come with free-text radiology reports and only a few have expert ground truth.

It walks an agent through:

- **Extraction:** turning reports into soft labels with `present / absent / not_addressed`, a calibrated probability, and a supporting quote
- **Auditing:** multilingual keyword checks, disagreement between label sources, repeated-read spread, calibration
- **Fusion:** cell-wise averaging of independent label sources
- **OOF correction:** blending out-of-fold image-model predictions into labels for findings that reports tend to omit
- **Fair evaluation:** paired experiments, expert-set discipline, and what a score can and cannot show

## Origin

Distilled from the public discussions and released code of the
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
