"""Evaluation metrics — for targets.yaml gate decisions (docs/BENCHMARK.md, configs/targets.yaml).

Produces low-false-positive operating-point metrics from window/call scores. Pure stdlib
(reuses simulate.py metrics).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .simulate import roc_auc, partial_auc, roc_points


def threshold_at_fpr(scores: Sequence[float], labels: Sequence[int], fpr_target: float) -> float:
    """Lowest score threshold satisfying FPR ≤ fpr_target (→ maximizes recall)."""
    pos = sorted((s for s, y in zip(scores, labels) if y == 1))
    neg = sorted((s for s, y in zip(scores, labels) if y == 0), reverse=True)
    if not neg:
        return min(scores) if scores else 0.0
    k = int(fpr_target * len(neg))                 # allowed number of false positives
    # Just above the point passing the top-k false positives → exceed the k-th largest neg score
    thr = neg[k] if k < len(neg) else neg[-1]
    return thr + 1e-9


def recall_at_fpr(scores: Sequence[float], labels: Sequence[int], fpr_target: float) -> float:
    """Recall (=TPR) at the FPR≤target operating point."""
    thr = threshold_at_fpr(scores, labels, fpr_target)
    P = sum(labels)
    if P == 0:
        return 0.0
    tp = sum(1 for s, y in zip(scores, labels) if y == 1 and s >= thr)
    return tp / P


def ece(scores: Sequence[float], labels: Sequence[int], bins: int = 10) -> float:
    """Expected Calibration Error — how well probability predictions match observed frequency (lower is better)."""
    n = len(scores)
    if n == 0:
        return 0.0
    tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i in range(n) if (lo <= scores[i] < hi) or (b == bins - 1 and scores[i] == 1.0)]
        if not idx:
            continue
        conf = sum(scores[i] for i in idx) / len(idx)
        acc = sum(labels[i] for i in idx) / len(idx)
        tot += (len(idx) / n) * abs(conf - acc)
    return tot


def window_metrics(scores: Sequence[float], labels: Sequence[int]) -> Dict[str, float]:
    return {
        "auc": roc_auc(scores, labels),
        "pauc_1pct": partial_auc(scores, labels, 0.01),
        "pauc_0_1pct": partial_auc(scores, labels, 0.001),
        "recall_at_fpr_1pct": recall_at_fpr(scores, labels, 0.01),
        "recall_at_fpr_0_1pct": recall_at_fpr(scores, labels, 0.001),
        "ece": ece(scores, labels),
        "n": len(scores), "n_pos": sum(labels),
    }
