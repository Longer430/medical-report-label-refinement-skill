#!/usr/bin/env python3
"""Training-only site bias (SKILL Section 6).

The image model learns f(x); during training the loss sees f(x) + b[site, finding],
so the per-site reporting habit is absorbed by b instead of the image path. At
inference b is dropped and only f(x) is used. Several RSNA 2026 Knee teams gained
about +0.006 to +0.007 private AUC with this, independently.

Pieces:
  build_site_keys(meta, cols, min_count)  -> site id per study (small groups merged)
  init_site_bias(labels, site_ids)        -> logit(group mean) - logit(global mean), numpy
  SiteBias (torch.nn.Module)              -> adds b[site] to logits; optional Gaussian prior
  param_groups(model, site_bias, ...)     -> separate LR / no weight decay for b

Typical training step:
    logits = model(x)                                   # (B, F)
    loss = bce_with_weights(site_bias(logits, site), y, w) + site_bias.penalty()
At validation / inference: use `logits` only (never add the bias).

Validation note: pooled OOF AUC against report labels usually goes DOWN with the
bias, because the report labels contain the habit. Track within-site AUC instead
(`label_tools.py within-site`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_site_keys(meta: pd.DataFrame, cols: list[str], min_count: int = 30,
                    other: str = "__other__") -> pd.Series:
    """Join metadata columns (e.g. report language, manufacturer, model, field
    strength) into one site key per study; groups smaller than min_count share
    one key. Normalise spellings (e.g. 'SIEMENS' vs 'Siemens Healthineers') first."""
    key = meta[cols].astype(str).apply(lambda r: "|".join(v.strip().lower() for v in r), axis=1)
    counts = key.value_counts()
    small = counts[counts < min_count].index
    return key.where(~key.isin(small), other)


def _logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def init_site_bias(labels: np.ndarray, site_ids: np.ndarray, n_sites: int | None = None) -> np.ndarray:
    """labels: (N, F) soft labels in [0, 1] (NaN = missing); site_ids: (N,) ints.
    Returns (S, F): logit(site mean) - logit(global mean), 0 where undefined."""
    labels = np.asarray(labels, float)
    site_ids = np.asarray(site_ids, int)
    S = n_sites or int(site_ids.max()) + 1
    glob = np.nanmean(labels, axis=0)
    out = np.zeros((S, labels.shape[1]))
    for s in range(S):
        m = site_ids == s
        if m.any():
            mean = np.nanmean(labels[m], axis=0)
            out[s] = np.where(np.isfinite(mean), _logit(mean) - _logit(glob), 0.0)
    return out


try:
    import torch
    import torch.nn as nn

    class SiteBias(nn.Module):
        """Adds a learned per-site, per-finding logit offset during training only.

        prior_sd: if set, penalty() returns sum(b^2) / (2 * prior_sd^2) / n_rows,
        a Gaussian (random-effect style) prior; 33rd place used sd = 0.5.
        """

        def __init__(self, n_sites: int, n_findings: int, init: np.ndarray | None = None,
                     prior_sd: float | None = None):
            super().__init__()
            b = torch.zeros(n_sites, n_findings) if init is None else torch.as_tensor(init, dtype=torch.float32)
            self.b = nn.Parameter(b.clone())
            self.prior_sd = prior_sd

        def forward(self, logits: "torch.Tensor", site: "torch.Tensor") -> "torch.Tensor":
            if not self.training:
                return logits  # never applied outside training
            return logits + self.b[site]

        def penalty(self) -> "torch.Tensor":
            if self.prior_sd is None:
                return self.b.new_zeros(())
            return (self.b ** 2).sum() / (2 * self.prior_sd ** 2) / self.b.shape[0]

    def param_groups(model: "nn.Module", site_bias: "SiteBias", lr: float = 1e-4,
                     bias_lr: float = 1e-2, weight_decay: float = 0.05):
        """AdamW groups: model as usual, site bias with its own LR and no decay
        (72nd place: lr 1e-2; 38th place: 5e-3)."""
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr,
                 "weight_decay": weight_decay},
                {"params": [site_bias.b], "lr": bias_lr, "weight_decay": 0.0}]

except ImportError:  # numpy helpers above still work without torch
    SiteBias = None
    param_groups = None
