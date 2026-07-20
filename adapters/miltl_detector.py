"""MiLTL as a bench 'row' — head-to-head with SOTA on the same bundle and same report format (see docs/BASELINES.md, docs/BENCHMARK.md).

MiLTL is a **call-level scorer**. A bench item is turned into a nibble CallStream, then scored by the
System-1 call decision (`MultiScaleCNNAdaptor` — multi-scale CNN over continuous/cumulative representations,
validation AUROC 0.925; see design notes).
- If `call.stream` is inline (pre-featurized), it is **used as-is** (real PEINN nibbles from DGX).
- Otherwise `call.transcript` is tiled into 8-second proxies -> text featurize -> CallStream (dev: Mock, DGX: PEINN injection).
Also works on text-only bundles (KorCCViD, cross-corpus): text modality when no wave. **No torch required (pure numpy)**.

modality:
- "text"  — text-only CNN (legacy comparison, KorCCViD).
- "wave"  — waveform-only CNN.
- "dual"  — two per-channel CNNs with late-fusion (P0; see design notes). wave_model is trained only on
            calls with waveform present; scoring uses channel gating (w_text-weighted sum when waveform
            exists, text-only fallback otherwise). Text-only bundles like KorCCViD automatically fall back
            to text_only -> waveform contribution is only measured on KorMMP (real audio).

This adapter is what lets `run_baselines.py` line MiLTL up against KoBERT/CNN/Tree/LLM on the **same bundle
and same metrics** in one row. Fair comparison on both Track1 (official split, in-dist) and Track2
(cross-corpus robustness Δ); see docs/BENCHMARK.md.

Real PEINN injection on DGX (kwargs = literals):
  python scripts/run_baselines.py --bundle bench_T0.jsonl \
     --detectors "adapters.miltl_detector:MiLTLDetector(text_adapter='myproj.peinn:score')"
With a stream-inline bundle (pre-featurized) no text_adapter is needed — nibbles are used directly.
"""
from __future__ import annotations

from typing import FrozenSet, List, Optional, Sequence

from miltl.baseline.detector import BaselineDetector
from miltl.nibble import MultiScaleCNNAdaptor, FeatureContext, NibbleThresholds
from miltl.nibble.featurize import MockTextFeaturizer, PeinnTextFeaturizer
from miltl.nibble.integrations import assemble_multimodal_stream, tile_text, load_callable
from miltl.nibble.schema import CallStream
from miltl.nibble.tife import TIFE


def _bits_tife(nibble: int) -> TIFE:
    """Lift a binary nibble to a {0,1} continuous TIFE (unfolds the 4 bits without information loss). Fallback when continuous tife is absent."""
    return TIFE((nibble >> 3) & 1, (nibble >> 2) & 1, (nibble >> 1) & 1, nibble & 1)


def _has_wave(cs: CallStream) -> bool:
    """Waveform channel present = at least one segment with wave_tife (False for text-only KorCCViD)."""
    return any(s.wave_tife is not None for s in cs.segments)


def _ensure_continuous(cs: CallStream) -> CallStream:
    """The CNN reads continuous text_tife/wave_tife (see design notes). If absent but a nibble exists, fill via bit-lift.
    Real PEINN (with_raw) streams already carry continuous values -> unchanged. Nibble-only streams fall back to {0,1} (≈ binary performance)."""
    for s in cs.segments:
        if s.text_tife is None and s.text_nibble is not None:
            s.text_tife = _bits_tife(s.text_nibble)
        if s.wave_tife is None and s.wave_nibble is not None:
            s.wave_tife = _bits_tife(s.wave_nibble)
    return cs


class MiLTLDetector(BaselineDetector):
    """MiLTL System-1 call scorer (built for always-on edge operation). Gate-2 (SLM) is a separate PoC (scripts/gate2_poc.py)."""

    name = "MiLTL"
    family = "multimodal"
    needs: FrozenSet[str] = frozenset({"nibble"})   # nibble source = inline stream OR featurizable transcript
    repro = "ok"
    notes = "System-1 통화 CNN(연속+누적). stream 인라인 우선, 없으면 transcript featurize."

    def __init__(self, text_adapter: Optional[str] = None, modality: str = "text",
                 words_per_seg: int = 14, epochs: int = 60, seed: int = 0,
                 kernels: Sequence[int] = (2, 3, 5), K: int = 12, L: int = 15,
                 encoding: str = "continuous", groups: Sequence[str] = ("seg", "cum"),
                 dump_nibbles: bool = False, bench: str = "", max_segments: int = 0,
                 w_text: float = 0.6):
        # Text featurizer: real PEINN (score_fn) when a spec is injected, otherwise the dev Mock.
        if text_adapter:
            self._textf = PeinnTextFeaturizer(load_callable(text_adapter))
        else:
            self._textf = MockTextFeaturizer()
        self.modality = modality
        if modality != "text":               # Distinguish report rows (text = default "MiLTL"; dual/wave are explicit)
            self.name = f"MiLTL({modality})"
        self.words_per_seg = words_per_seg
        self.w_text = w_text                    # dual late-fusion weight (see docs/ARCHITECTURE.md, hybrid); falls back when no waveform
        self.L = L                              # Observation-window segment count (= nibbles). D1-ⓐ: front observation budget of 26 (= 360 words / 14).
        # encoding (saturation relief: zscore/quantile) · groups (waveform trajectory: +delta) — must match between train and inference (checkpoint bundled).
        _mk = lambda m: MultiScaleCNNAdaptor(
            kernels=tuple(kernels), K=K, L=L, encoding=encoding, groups=tuple(groups),
            epochs=epochs, seed=seed, modality=m)
        if modality == "dual":                  # Per-channel adapters (late-fusion, channel gating)
            self._text_model, self._wave_model = _mk("text"), _mk("wave")
            self._wave_fitted = False
            self._model = None
        else:
            self._model = _mk(modality)
        self._fitted = False
        self.dump_nibbles = dump_nibbles        # __nibbles.jsonl collection (see design notes)
        self.bench = bench
        self.max_segments = max_segments        # Length-control clipping (see design notes): first N segments only (0 = unlimited)
        self.nibble_log: List[dict] = []

    # ---- Bench item -> CallStream (inline stream takes priority) ----------
    def _clip(self, cs: Optional[CallStream]) -> Optional[CallStream]:
        """Length control (see design notes): clip to the first max_segments segments — long calls only up to the same window (diversity preserved)."""
        if cs is None or not self.max_segments or len(cs.segments) <= self.max_segments:
            return cs
        return CallStream(cs.call_id, cs.source, cs.label, cs.segments[:self.max_segments])

    def _to_stream(self, call) -> Optional[CallStream]:
        if getattr(call, "stream", None) is not None:
            return self._clip(_ensure_continuous(call.stream))
        text = getattr(call, "transcript", None)
        if not text:
            return None
        segs = tile_text([text], self.words_per_seg)
        if not segs:
            return None
        return self._clip(_ensure_continuous(assemble_multimodal_stream(
            call_id=str(getattr(call, "call_id", "c")),
            source=str(getattr(call, "source", "?")),
            label=("phishing" if int(getattr(call, "label", 0)) == 1 else "benign"),
            text_segments=segs, text_featurizer=self._textf,
        )))

    # ---- Training (call CNN) ---------------------------------------------
    def fit(self, train_calls: Sequence) -> None:
        import sys, time
        calls: List[CallStream] = []
        labels: List[int] = []
        t0 = time.time()
        n = len(train_calls)
        for i, c in enumerate(train_calls, 1):
            cs = self._to_stream(c)                          # PEINN featurize (can be slow)
            if cs is None:
                continue
            calls.append(cs)
            labels.append(int(getattr(c, "label", 0)))
            if i % 100 == 0:
                el = time.time() - t0
                print(f"[MiLTL] featurize {i}/{n} · {el:.0f}s · {1000*el/i:.0f}ms/call",
                      file=sys.stderr, flush=True)
        print(f"[MiLTL] featurize 완료 {len(calls)}건 · CNN 학습 시작", file=sys.stderr, flush=True)
        if len(set(labels)) < 2:            # Single class -> cannot train -> untrained (0.5)
            self._fitted = False
            return
        if self.modality == "dual":
            self._text_model.fit(calls, labels, FeatureContext.fit(calls, NibbleThresholds(), "text"))
            # Waveform: only calls with wave_tife (text-only sources like KorCCViD skip wave training -> text_only fallback)
            wpair = [(cs, l) for cs, l in zip(calls, labels) if _has_wave(cs)]
            if len(wpair) >= 10 and len({l for _, l in wpair}) >= 2:
                wc = [c for c, _ in wpair]; wl = [l for _, l in wpair]
                self._wave_model.fit(wc, wl, FeatureContext.fit(wc, NibbleThresholds(), "wave"))
                self._wave_fitted = True
            print(f"[MiLTL] dual — 파형 학습 {'O('+str(len(wpair))+')' if self._wave_fitted else 'X(text_only)'}",
                  file=sys.stderr, flush=True)
        else:
            ctx = FeatureContext.fit(calls, NibbleThresholds(), self.modality)
            self._model.fit(calls, labels, ctx)
        self._fitted = True

    def _score_dual(self, cs) -> float:
        """late-fusion: channel gating — only present channels; if both, w_text-weighted sum (hybrid.py style)."""
        s_t = self._text_model.score_call(cs)
        s_w = self._wave_model.score_call(cs) if (self._wave_fitted and _has_wave(cs)) else None
        if s_w is None:
            return s_t                       # Waveform missing (KorCCViD) -> text-only fallback
        return self.w_text * s_t + (1.0 - self.w_text) * s_w

    # ---- Call decision -----------------------------------------------------
    def score(self, call) -> float:
        if not self._fitted:
            return 0.5
        cs = self._to_stream(call)
        if cs is None:
            return 0.5
        try:
            s = float(self._score_dual(cs) if self.modality == "dual" else self._model.score_call(cs))
        except Exception:                    # e.g. representation too short -> neutral
            s = 0.5
        if self.dump_nibbles:
            self.nibble_log.append(self._nibble_record(call, cs, s))
        return s

    def _nibble_record(self, call, cs, score) -> dict:
        """Per-case PEINN raw (__nibbles.jsonl; see design notes) — continuous tife + nibble + occupancy."""
        def _bits(n): return ((n >> 3) & 1, (n >> 2) & 1, (n >> 1) & 1, n & 1)
        tn, wn = cs.nibble_streams()
        tt = [[round(s.text_tife.T, 4), round(s.text_tife.I, 4), round(s.text_tife.F, 4),
               round(s.text_tife.E, 4)] if s.text_tife else None for s in cs.segments]
        wt = [[round(s.wave_tife.T, 4), round(s.wave_tife.I, 4), round(s.wave_tife.F, 4),
               round(s.wave_tife.E, 4)] if s.wave_tife else None for s in cs.segments]

        def occ(seq):
            v = [n for n in seq if n is not None]
            if not v: return {}
            b = [_bits(n) for n in v]
            return {k: round(sum(x[i] for x in b) / len(b), 4) for i, k in enumerate("TIFE")}
        return {"bench": self.bench, "detector": self.name, "case_id": str(getattr(call, "call_id", "?")),
                "label": int(getattr(call, "label", 0)), "score": round(score, 5),
                "text_nibbles": [n for n in tn if n is not None],
                "wave_nibbles": [n for n in wn if n is not None],
                "text_tife": tt, "wave_tife": wt,
                "occ_text": occ(tn), "occ_wave": occ(wn)}
