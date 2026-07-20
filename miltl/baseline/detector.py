"""BaselineDetector contract + unified metrics + runner (docs/BASELINES.md).

Every detector passes through the same compute_metrics and produces the same ResultRow.
Time (latency) is wall-clock but does not affect reproducibility (informational column in
the result sheet). Metric computation is pure stdlib.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence

from miltl.nibble.evaluate import window_metrics, threshold_at_fpr
from miltl.nibble.simulate import roc_auc


class BaselineDetector:
    """Subclasses define name/family/needs and implement score() (plus optional fit())."""

    name: str = "base"
    family: str = "text"                       # llm | text | audio | multimodal
    needs: FrozenSet[str] = frozenset({"text"})
    repro: str = "ok"                          # ok | partial | paper-only
    notes: str = ""

    def fit(self, train_calls: Sequence) -> None:
        """Supervised training (optional). Training-free detectors do not override."""
        return None

    def score(self, call) -> float:
        raise NotImplementedError


def _best_f1_threshold(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Threshold maximizing F1 (candidates = each score). For selecting the train operating point."""
    cand = sorted(set(scores))
    best_t, best_f1 = 0.5, -1.0
    P = sum(labels)
    for t in cand:
        tp = sum(1 for s, y in zip(scores, labels) if s >= t and y == 1)
        fp = sum(1 for s, y in zip(scores, labels) if s >= t and y == 0)
        if tp == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / P if P else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def _point_metrics(scores, labels, thr) -> Dict[str, float]:
    tp = sum(1 for s, y in zip(scores, labels) if s >= thr and y == 1)
    fp = sum(1 for s, y in zip(scores, labels) if s >= thr and y == 0)
    fn = sum(1 for s, y in zip(scores, labels) if s < thr and y == 1)
    tn = sum(1 for s, y in zip(scores, labels) if s < thr and y == 0)
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / n if n else 0.0
    return {"f1": f1, "accuracy": acc, "precision": prec, "recall": rec}


def compute_metrics(
    scores: Sequence[float], labels: Sequence[int], thr: Optional[float] = None,
) -> Dict[str, float]:
    """Unified metrics. thr (operating threshold) should be the value chosen on train (fallback: test-best-F1, informational only)."""
    wm = window_metrics(scores, labels)                 # auc·pauc·recall@fpr·ece·n·n_pos
    if thr is None:
        thr = _best_f1_threshold(scores, labels)
    pm = _point_metrics(scores, labels, thr)
    return {
        "auroc": wm["auc"], "pauc_1pct": wm["pauc_1pct"],
        "recall_at_fpr_1pct": wm["recall_at_fpr_1pct"],
        "recall_at_fpr_0_1pct": wm["recall_at_fpr_0_1pct"],
        "f1": pm["f1"], "accuracy": pm["accuracy"],
        "precision": pm["precision"], "recall": pm["recall"],
        "ece": wm["ece"], "op_threshold": thr,
        "pr_auc": _pr_auc(scores, labels),          # primary KPI under 2:8 imbalance
        "n": wm["n"], "n_pos": wm["n_pos"],
    }


def _pr_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Precision-Recall AUC (average precision). More sensitive than ROC under imbalance (2:8). Pure stdlib."""
    P = sum(labels)
    if P == 0 or P == len(labels):
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = fp = 0
    ap = 0.0
    prev_recall = 0.0
    for i in order:
        if labels[i] == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / P
        prec = tp / (tp + fp)
        ap += prec * (recall - prev_recall)         # precision weighted by ΔRecall
        prev_recall = recall
    return ap


@dataclass
class PerCase:
    """Per-case decision record (docs/BENCHMARK.md __cases.csv) — for dense stats and qualitative analysis."""
    case_id: str
    source: str
    label: int
    score: float
    latency_ms: float
    scenario_type: str = "?"
    modality: str = "?"
    fmt: str = "?"
    n_words: int = 0
    n_segments: int = 0
    duration_s: float = 0.0
    order_idx: int = -1
    wer: float = -1.0            # ASR WER vs GT transcript (-1 if absent) — analyzes text-quality impact
    snr_db: float = -999.0       # source audio SNR (dB, meta) — channel quality
    n_speakers: int = 0          # speaker count (meta)
    tier: str = "?"              # harm | hardneg | diverse (ORR slicing)


@dataclass
class ResultRow:
    name: str
    family: str
    modality: str
    repro: str
    metrics: Dict[str, float] = field(default_factory=dict)
    n_test: int = 0
    n_pos: int = 0
    skipped: int = 0
    latency_ms: float = 0.0
    eval_mode: str = "in-dist"                 # "in-dist" (same-source split) | "xcorpus:A→B" (cross-source)
    notes: str = ""
    error: Optional[str] = None
    cases: List["PerCase"] = field(default_factory=list)   # for the dense result sheet (docs/BENCHMARK.md)

    def to_dict(self) -> Dict:
        return {
            "name": self.name, "family": self.family, "modality": self.modality,
            "repro": self.repro, "eval_mode": self.eval_mode,
            "n_test": self.n_test, "n_pos": self.n_pos,
            "skipped": self.skipped, "latency_ms": round(self.latency_ms, 2),
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "notes": self.notes, "error": self.error,
        }


def _percase(c, score: float, latency_ms: float, order_idx: int) -> "PerCase":
    """BenchmarkCall → PerCase (dense meta). meta takes priority, else derived from fields."""
    meta = getattr(c, "meta", {}) or {}
    st = getattr(c, "stream", None)
    audio = getattr(c, "audio_uri", None)
    tr = getattr(c, "transcript", None)
    n_seg = len(st.segments) if st is not None else 0
    n_words = meta.get("n_words") or (len(tr.split()) if tr else 0)
    modality = meta.get("modality") or (
        "dual" if (audio and (tr or st)) else "wave" if audio else "stream" if st else "text")
    fmt = meta.get("format") or (
        (audio.rsplit(".", 1)[-1] if audio and "." in audio else "audio") if audio else "text")
    return PerCase(
        case_id=str(getattr(c, "call_id", "?")), source=str(getattr(c, "source", "?")),
        label=int(getattr(c, "label", 0)), score=score, latency_ms=latency_ms,
        scenario_type=meta.get("scenario_type", "?"), modality=modality, fmt=fmt,
        n_words=int(n_words), n_segments=n_seg,
        duration_s=float(meta.get("duration_s", 0.0)), order_idx=order_idx,
        wer=float(meta.get("wer", -1.0)), snr_db=float(meta.get("snr_db", -999.0)),
        n_speakers=int(meta.get("n_speakers", 0)), tier=str(meta.get("tier", "?")))


def run_benchmark(detector: BaselineDetector, calls: Sequence,
                  test_calls: Optional[Sequence] = None) -> ResultRow:
    """Run one detector through the frozen benchmark. Calls missing required fields are skipped (explicit). Failures are isolated as error.

    test_calls=None: same-bundle split (in-dist). test_calls given: cross-corpus — fit on all of
    `calls`, evaluate on all of `test_calls` (train source≠test source → measures generalization
    and shortcut collapse; fairness per docs/BASELINES.md).
    """
    modality = "+".join(sorted(detector.needs))
    row = ResultRow(name=detector.name, family=detector.family,
                    modality=modality, repro=detector.repro, notes=detector.notes)
    try:
        usable = [c for c in calls if c.consumable(detector.needs)]
        row.skipped = len(calls) - len(usable)
        if test_calls is None:
            train = [c for c in usable if c.split == "train"]
            test = [c for c in usable if c.split == "test"]
        else:                                          # cross-corpus: calls=full train, test_calls=full test
            train = usable
            test = [c for c in test_calls if c.consumable(detector.needs)]
            row.eval_mode = "xcorpus"
        if not test:
            row.error = "no consumable test calls"
            return row

        detector.fit(train)

        scores, labels, t0 = [], [], time.perf_counter()
        for i, c in enumerate(test):
            tc = time.perf_counter()
            s = float(detector.score(c))
            lat = 1000.0 * (time.perf_counter() - tc)
            scores.append(s)
            labels.append(int(c.label))
            row.cases.append(_percase(c, s, lat, i))       # dense record (docs/BENCHMARK.md)
        dt = time.perf_counter() - t0
        row.latency_ms = 1000.0 * dt / max(1, len(test))

        # Operating threshold is chosen on train (no test re-tuning). If train is insufficient, test-best-F1 (informational).
        thr = None
        if train:
            tr_s = [float(detector.score(c)) for c in train]
            tr_y = [int(c.label) for c in train]
            if sum(tr_y) and (len(tr_y) - sum(tr_y)):
                thr = _best_f1_threshold(tr_s, tr_y)
        row.metrics = compute_metrics(scores, labels, thr)
        row.n_test = len(test)
        row.n_pos = sum(labels)
    except Exception as e:  # noqa: BLE001 — isolated so one detector's failure does not block the whole result sheet
        row.error = f"{type(e).__name__}: {e}"
    return row
