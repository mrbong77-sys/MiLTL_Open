"""Wave Seq detector + text⊕wave fusion (approach B; see docs/BASELINES.md) — for inference and bench.

- `WaveSeqDetector`: audio_uri -> prosody sequence -> CNN (score_matrix). 0 when no audio (inactive).
- `MiLTLDualDetector`: text MiLTL-Seq ⊕ wave — **normalized-max OR** fusion (aimed at recall):
    score = max(text/text_thr, wave/wave_thr); text only when no audio. Threshold 1.0 = either exceeds its own threshold.
    -> Harm the text missed (garbled ASR) is recovered when wave catches it. Calls without audio (callcenter) fall back to text.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import FrozenSet, Optional

import numpy as np

from miltl.baseline.detector import BaselineDetector
from miltl.nibble.seq_adaptor import MultiScaleCNNAdaptor
from miltl.nibble.wave_seq import audio_to_prosody, prosody_vecs, seq_matrix


def _audio_uri(call) -> Optional[str]:
    au = getattr(call, "audio_uri", None)
    if au:
        return au
    m = getattr(call, "meta", None)
    return (m or {}).get("audio_uri") if isinstance(m, dict) else None


class WaveSeqDetector(BaselineDetector):
    """Wave prosody-sequence CNN — frozen load. Scores only calls with audio (0 otherwise)."""

    family = "wave"
    needs: FrozenSet[str] = frozenset({"audio"})
    repro = "ok"

    def __init__(self, ckpt: str = "artifacts/models/wave_seq_fss.json", name: str = "Wave-Seq",
                 codec_equalize: bool = False):
        self.name = name
        self._codec_eq = codec_equalize                      # Fair audio-channel equalization (see docs/BENCHMARK.md)
        d = json.loads(Path(ckpt).read_text(encoding="utf-8"))
        self.adaptor = MultiScaleCNNAdaptor.from_dict(d["adaptor"])
        self.mean = np.asarray(d["norm"]["mean"], np.float32)
        self.std = np.asarray(d["norm"]["std"], np.float32)
        self.L = int(d["config"]["L"])
        self.intrinsic_op_thr = float(d["op_thr"])
        self.intrinsic_src = f"고정(wave held-out best-F1={self.intrinsic_op_thr:.4f})"

    def fit(self, train_calls) -> None:
        return None

    def score(self, call) -> float:
        au = _audio_uri(call)
        if not au:
            return 0.0                                       # No audio -> wave inactive
        try:
            prosody = audio_to_prosody(au, budget_segs=self.L, codec_equalize=self._codec_eq)
        except Exception:  # noqa: BLE001
            return 0.0
        if len(prosody) < 3:
            return 0.0
        M = seq_matrix(prosody_vecs(prosody), self.mean, self.std, L=self.L)
        return float(self.adaptor.score_matrix(M))


class MiLTLDualDetector(BaselineDetector):
    """MiLTL text⊕wave fusion (normalized-max OR). Recovers harm the text missed via wave (recall). Text fallback when no audio."""

    family = "multimodal"
    needs: FrozenSet[str] = frozenset({"text"})
    repro = "ok"

    def __init__(self, text_ckpt: str = "artifacts/models/miltl_seq_korccvid.json",
                 wave_ckpt: str = "artifacts/models/wave_seq_fss.json",
                 name: str = "MiLTL-Dual(text+wave)", codec_equalize: bool = False):
        from adapters.baselines.miltl_seq import MiLTLSeqDetector
        self.name = name
        self._text = MiLTLSeqDetector(ckpt=text_ckpt)
        self._wave = WaveSeqDetector(ckpt=wave_ckpt, codec_equalize=codec_equalize)
        self._t_thr = max(self._text.intrinsic_op_thr, 1e-9)
        self._w_thr = max(self._wave.intrinsic_op_thr, 1e-9)
        # Normalized-max OR: score/thr >= 1 = exceeds its own threshold. max >= 1 = either one is harm -> bench threshold 1.0.
        self.intrinsic_op_thr = 1.0
        self.intrinsic_src = "고정(정규화-max OR·text|wave 각 자기임계, recall 융합)"

    def fit(self, train_calls) -> None:
        return None

    def score(self, call) -> float:
        ts = float(self._text.score(call)) / self._t_thr
        if _audio_uri(call) is None:
            return ts                                        # No audio (callcenter) -> text only
        ws = float(self._wave.score(call)) / self._w_thr
        return max(ts, ws)                                   # OR: either exceeds its own threshold -> harm
