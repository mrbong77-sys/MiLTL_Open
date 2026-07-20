"""Learned MiLTL nibble-sequence classifier — inference adapter (bench-mounted) — see design notes.

Loads the checkpoint (weights + threshold + config) trained on KorCCViD by `scripts/train_miltl_seq.py`
and lines it up as a bench row. Unlike the static heads (F/margin) and the Kalman gate, this is a CNN
**supervised on the continuous nibble T/I/F/E trajectories** (the user-confirmed approach).

Frozen contract: weights and threshold are both fixed on **korccvid** (train-set held-out best-F1) -> the
bench uses the **intrinsic threshold** (`intrinsic_op_thr`, treated the same as Kalman/B3). No tuning of
any kind on KorMMP.

Observation/decision: only the first obs_words (360) are observed (same window as bench --obs-words).
**Calls below the anchor (15 nibbles = 210 words) = undecidable -> 0.0 (safe)** = leak (FN) if harm —
MiLTL's delay-cost principle (same as Kalman).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import FrozenSet, Optional

from miltl.baseline.detector import BaselineDetector
from miltl.nibble.seq_adaptor import MultiScaleCNNAdaptor


class MiLTLSeqDetector(BaselineDetector):
    """Learned nibble-sequence CNN — frozen load, inference only. score = CNN harm probability (continuous). Intrinsic threshold bundled."""

    family = "text"
    needs: FrozenSet[str] = frozenset({"text"})
    repro = "ok"

    def __init__(self, ckpt: str = "artifacts/models/miltl_seq_korccvid.json",
                 text_adapter: Optional[str] = None, name: str = "MiLTL-Seq(learned)"):
        from adapters.miltl_detector import MiLTLDetector
        self.name = name
        d = json.loads(Path(ckpt).read_text(encoding="utf-8"))
        cfg = d["config"]
        self.seg_words = int(cfg["seg_words"])
        self.anchor_words = int(cfg["anchor"]) * self.seg_words        # 210: below = undecidable (FN)
        self.L = int(cfg["L"])
        # Intrinsic threshold = **korccvid held-out best-F1** (frozen on its own training distribution, bench-independent).
        # Measured (v3): held-out 0.0061 nearly matched the kormmp-optimal 0.0088 -> robust transfer (F1 0.800). The sigmoid
        # is uncalibrated, so a fixed 0.5 drifts across versions (F1 0.55 under v3 score compression) -> held-out matches
        # the model's actual output scale and is correct.
        self.intrinsic_op_thr = float(d["op_thr"])
        self.intrinsic_src = f"고정(korccvid held-out best-F1={self.intrinsic_op_thr:.4f}, 학습시 F1 {d.get('train',{}).get('held_f1','?')})"
        # featurizer: reuse the training-time adapter from the checkpoint config (currently KoEngine) by default (inference = training featurize).
        ta = text_adapter or cfg.get("text_adapter")
        self._inner = MiLTLDetector(text_adapter=ta, modality="text",
                                    words_per_seg=self.seg_words, L=self.L, max_segments=self.L)
        self._inner._model = MultiScaleCNNAdaptor.from_dict(d["adaptor"])   # Inject the trained CNN
        self._inner._fitted = True
        self.notes = (f"학습형 니블 CNN(L={self.L}, seg {self.seg_words}단어). 앵커 {self.anchor_words}단어 "
                      f"미달=판정불가→safe(harm이면 FN). 내재임계 {self.intrinsic_op_thr:.4f}[korccvid].")

    def fit(self, train_calls) -> None:
        return None                                                    # Frozen (no retraining)

    def score(self, call) -> float:
        # Below anchor (observed words < 210) = not ready to decide -> 0.0 (safe). If harm, leak = FN (honest delay cost).
        tr = getattr(call, "transcript", "") or ""
        if len(tr.split()) < self.anchor_words:
            return 0.0
        return float(self._inner.score(call))
