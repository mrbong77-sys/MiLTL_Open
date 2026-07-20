"""Threshold simulation harness — precisely pre-determines the on/off threshold for each bit.

Core question: where should each T/I/F/E channel threshold τ be placed so that **nibble patterns
accumulated over a 1–2 minute window** best separate benign vs phishing? (Simulated on top of
continuous PEINN outputs, without rebuilding a nibble corpus.)

Provided functionality:
  - Separability metrics: ROC-AUC, partial AUC@FPR≤budget (aligned with targets.yaml operating points).
  - Per-channel information: **mutual information (MI)** between bit and call label — as a function
    of τ (how much information each bit carries about the label).
  - **Jensen–Shannon divergence** between class nibble-state distributions (pattern separability,
    classifier-agnostic).
  - **Window construction** (1–2 min): cut the continuous PEINN stream with sliding windows,
    inheriting the call label (no new labeling).
  - **Coordinate-ascent search**: start from a seed (τ_v2.1) and maximize window separability
    (pAUC) under constraints (escalation budget).

Pure stdlib. Window features/encoding reuse encoder/features/gate1 as-is.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .encoder import NibbleEncoder, NibbleThresholds, unpack
from .features import extract_features
from .gate1 import Gate1Scorer, RuleLogisticGate1
from .synth import SynthCall
from .tife import TIFE


# ============================================================ Metrics =========

def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Mann–Whitney U based AUC (ties get average rank). labels ∈ {0,1}."""
    pairs = sorted(zip(scores, labels), key=lambda z: z[0])
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0            # 1-based average rank
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    sum_pos = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def roc_points(scores: Sequence[float], labels: Sequence[int]) -> List[Tuple[float, float]]:
    """(FPR, TPR) point sequence — threshold sweep in descending score order."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    P = sum(labels)
    N = len(labels) - P
    if P == 0 or N == 0:
        return [(0.0, 0.0), (1.0, 1.0)]
    tp = fp = 0
    pts = [(0.0, 0.0)]
    prev = None
    for i in order:
        if scores[i] != prev and prev is not None:
            pts.append((fp / N, tp / P))
        if labels[i] == 1:
            tp += 1
        else:
            fp += 1
        prev = scores[i]
    pts.append((fp / N, tp / P))
    return pts


def partial_auc(scores: Sequence[float], labels: Sequence[int], fpr_max: float = 0.01) -> float:
    """Normalized partial AUC over FPR∈[0,fpr_max] (McClish). Low-false-positive operating-point metric."""
    pts = roc_points(scores, labels)
    area = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x1 <= fpr_max:
            area += (x1 - x0) * (y0 + y1) / 2.0
        elif x0 < fpr_max < x1:                # interpolate at the boundary, then truncate
            t = (fpr_max - x0) / (x1 - x0)
            y_mid = y0 + t * (y1 - y0)
            area += (fpr_max - x0) * (y0 + y_mid) / 2.0
            break
        else:
            break
    return area / fpr_max if fpr_max > 0 else 0.0


def mutual_info_bit(bits: Sequence[int], labels: Sequence[int]) -> float:
    """Mutual information I(B;Y) [bits] between binary bit B and binary label Y."""
    n = len(bits)
    if n == 0:
        return 0.0
    joint: Dict[Tuple[int, int], int] = {}
    pb = [0, 0]
    py = [0, 0]
    for b, y in zip(bits, labels):
        joint[(b, y)] = joint.get((b, y), 0) + 1
        pb[b] += 1
        py[y] += 1
    mi = 0.0
    for (b, y), c in joint.items():
        pxy = c / n
        px = pb[b] / n
        p_y = py[y] / n
        if pxy > 0 and px > 0 and p_y > 0:
            mi += pxy * math.log2(pxy / (px * p_y))
    return mi


def _kl(p: Sequence[float], q: Sequence[float]) -> float:
    s = 0.0
    for pi, qi in zip(p, q):
        if pi > 0 and qi > 0:
            s += pi * math.log2(pi / qi)
    return s


def js_divergence_hist(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen–Shannon divergence [0,1] (base 2) — separability of two nibble-state distributions."""
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


# ==================================================== Window construction =========

@dataclass
class Window:
    tife: List[TIFE]
    label: int


def make_windows(calls: List[SynthCall], win: int, stride: int, min_len: int = 8) -> List[Window]:
    """Cut the continuous PEINN stream into sliding windows (win segments ≈ 1–2 min), inheriting the call label.

    If a call is shorter than win, the whole call is one window. No new labeling (not a corpus rebuild)."""
    out: List[Window] = []
    for c in calls:
        L = len(c)
        if L <= win:
            if L >= min_len:
                out.append(Window(list(c.tife), c.label))
            continue
        for s in range(0, L - win + 1, stride):
            out.append(Window(c.tife[s : s + win], c.label))
    return out


# ==================================================== Evaluation/search ===========

@dataclass
class SimReport:
    thresholds: NibbleThresholds
    auc: float
    pauc: float
    js: float
    escalation_rate: float
    per_bit_mi: Dict[str, float]
    state_hist: Dict[int, List[float]]      # label → 16-state normalized histogram


def _window_scores(windows: List[Window], enc: NibbleEncoder, scorer: Gate1Scorer):
    scores, labels = [], []
    for w in windows:
        feats = extract_features(enc.encode_stream(w.tife))
        scores.append(scorer.score(feats))
        labels.append(w.label)
    return scores, labels


def _class_state_hist(windows: List[Window], enc: NibbleEncoder) -> Dict[int, List[float]]:
    acc = {0: [0.0] * 16, 1: [0.0] * 16}
    cnt = {0: 0, 1: 0}
    for w in windows:
        for nib in enc.encode_stream(w.tife):
            acc[w.label][nib] += 1.0
            cnt[w.label] += 1
    for y in (0, 1):
        tot = cnt[y] or 1
        acc[y] = [x / tot for x in acc[y]]
    return acc


def _per_bit_mi(windows: List[Window], th: NibbleThresholds) -> Dict[str, float]:
    """Per-segment MI of each bit vs window label (conflict off, pure per-channel information)."""
    bT, bI, bF, bE, ys = [], [], [], [], []
    for w in windows:
        for x in w.tife:
            bT.append(1 if x.T >= th.tau_T else 0)
            bI.append(1 if x.I >= th.tau_I else 0)
            bF.append(1 if x.F >= th.tau_F else 0)
            bE.append(1 if x.E >= th.tau_E else 0)
            ys.append(w.label)
    return {
        "T": mutual_info_bit(bT, ys), "I": mutual_info_bit(bI, ys),
        "F": mutual_info_bit(bF, ys), "E": mutual_info_bit(bE, ys),
    }


def evaluate(
    windows: List[Window],
    th: NibbleThresholds,
    scorer: Optional[Gate1Scorer] = None,
    fpr_max: float = 0.01,
    escalate_band: Tuple[float, float] = (0.15, 0.85),
) -> SimReport:
    """Compute window separability, information content, and escalation rate at the given thresholds."""
    scorer = scorer or RuleLogisticGate1()
    enc = NibbleEncoder(th)
    scores, labels = _window_scores(windows, enc, scorer)
    lo, hi = escalate_band
    escal = sum(1 for s in scores if lo < s < hi) / len(scores) if scores else 0.0
    hist = _class_state_hist(windows, enc)
    return SimReport(
        thresholds=th,
        auc=roc_auc(scores, labels),
        pauc=partial_auc(scores, labels, fpr_max),
        js=js_divergence_hist(hist[0], hist[1]),
        escalation_rate=escal,
        per_bit_mi=_per_bit_mi(windows, th),
        state_hist=hist,
    )


# Candidate threshold grid per channel
DEFAULT_GRID = {
    "tau_T": [round(0.2 + 0.05 * i, 2) for i in range(13)],   # 0.20..0.80
    "tau_F": [round(0.2 + 0.05 * i, 2) for i in range(13)],
    "tau_I": [round(0.2 + 0.05 * i, 2) for i in range(13)],
    "tau_C": [round(0.15 + 0.05 * i, 2) for i in range(11)],  # 0.15..0.65
    "tau_E": [round(0.2 + 0.05 * i, 2) for i in range(13)],
}


def coordinate_search(
    windows: List[Window],
    seed_th: Optional[NibbleThresholds] = None,
    grid: Optional[Dict[str, List[float]]] = None,
    scorer: Optional[Gate1Scorer] = None,
    fpr_max: float = 0.01,
    escalation_budget: float = 0.20,
    penalty_lambda: float = 0.7,
    passes: int = 2,
    objective: str = "pauc",
) -> Tuple[NibbleThresholds, SimReport, List[Tuple[str, float, float]]]:
    """Coordinate ascent starting from the seed — sweep channels one at a time to maximize the objective.

    Objective = objective(pauc/auc) − λ·max(0, escalation_rate − budget).
    Exceeding the escalation budget is softly penalized (even if the seed itself exceeds it,
    the search moves toward the feasible region).
    Returns: best τ, report, search log (channel, chosen τ, objective value at that point).
    """
    grid = grid or DEFAULT_GRID
    scorer = scorer or RuleLogisticGate1()
    th = seed_th or NibbleThresholds()
    log: List[Tuple[str, float, float]] = []

    def obj(rep: SimReport) -> float:
        return getattr(rep, objective) - penalty_lambda * max(0.0, rep.escalation_rate - escalation_budget)

    best = evaluate(windows, th, scorer, fpr_max)
    for _ in range(passes):
        for name in ("tau_F", "tau_E", "tau_I", "tau_T", "tau_C"):
            cur_best_val = obj(best)
            cur_best_v = getattr(th, name)
            for v in grid[name]:
                cand = NibbleThresholds(**{**th.__dict__, name: v})
                rep = evaluate(windows, cand, scorer, fpr_max)
                if obj(rep) > cur_best_val + 1e-9:
                    cur_best_val, cur_best_v, best = obj(rep), v, rep
            setattr(th, name, cur_best_v)
            log.append((name, cur_best_v, cur_best_val))
    return th, best, log
