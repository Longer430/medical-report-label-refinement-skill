# Toy data (fictional)

Ten **invented** studies with made-up English and Spanish report sentences. No real patient, report or
competition data. They exist so the scripts can be run in seconds without downloading anything.

- `toy_reports.csv`: study ID + fictional report text
- `toy_labels_a.csv`: a reasonable soft-label table
- `toy_labels_b.csv`: the same, but it **misses synovitis in the Spanish reports** (the pattern found in the RSNA example)
- `toy_labels_c_with_gaps.csv`: a third source with ~20% missing cells, for `build_blend.py`

```bash
python scripts/audit_labels.py toy_data/toy_labels_a.csv toy_data/toy_labels_b.csv \
    --reports toy_data/toy_reports.csv --names a b --out audit_out
python scripts/build_blend.py toy_data/toy_labels_a.csv toy_data/toy_labels_c_with_gaps.csv audit_out/toy_blend.csv
```

Expected: the audit flags synovitis with 3 disagreements, all "a higher", all with a keyword in the report
(TOY-002, TOY-004, TOY-009). It also flags TOY-005 for Effusion: the report says "No effusion", so the keyword
matched a negation; it is listed last, as priority 4 ("keyword, both low"). That is why flagged cells are for review, never for automatic relabelling.
