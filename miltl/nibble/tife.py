"""TIFE — per-segment 4-channel signal (T, I, F, E) and its production adapter.

MiLTL treats T/I/F/E as **independent channels** (inheriting PEINN v2.1's 3-sigmoid neutro head
plus a separate energy signal, but not its routing structure). Each channel is an independent
signal normalized to [0,1]:

  T : benign confidence                — maps to the v2.1 neutro T-sigmoid
  I : ignorance (uncertainty)          — maps to the v2.1 neutro I-sigmoid
  F : phishing (harm) confidence       — maps to the v2.1 neutro F-sigmoid
  E : energy/urgency (prosody-bound)   — new MiLTL channel (bound to speech prosody). Different scale from v2.1 score_energy.

C (contradiction) is a derived value, not a head output: C = min(T, F) (independent sigmoids
mean T and F can fire simultaneously).

Note: because the sigmoids are independent, T+F+I does not sum to 1. (The simplex of
pea_eval evidential_head.opinion is a separate experiment.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol


@dataclass(frozen=True)
class TIFE:
    """4-channel signal of one segment. All in [0,1] (E after its own normalization)."""
    T: float
    I: float
    F: float
    E: float

    @property
    def C(self) -> float:
        """Contradiction — derived. Independent sigmoids let T and F fire together; their minimum is the conflict signal."""
        return min(self.T, self.F)

    def clamp(self) -> "TIFE":
        c = lambda x: 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)
        return TIFE(c(self.T), c(self.I), c(self.F), c(self.E))


class TIFEProvider(Protocol):
    """Adapter interface converting a segment (text + optional prosody) into a TIFE.

    The production implementation binds the PEINN v2.1 neutro head (T/I/F) plus the MiLTL
    prosody-energy extractor (E). This file only fixes the interface — heavy dependencies
    (torch/transformers) are imported only in the real implementation.
    """

    def segment(self, transcript: str, prosody: Optional[dict] = None) -> TIFE: ...


class MockTIFEProvider:
    """Deterministic mock for smoke tests/demos — builds a plausible TIFE from text hints + prosody.

    This is for pipeline validation, not a trained head. It imitates T/I/F/E via keywords and
    prosody. Deterministic (no randomness) and swappable for a real-inference adapter.
    """

    HARM_HINTS = ("대출", "수수료", "입금", "계좌", "신분증", "공증", "위임", "선지급", "송금", "카톡", "앱", "어플")
    PRESSURE_HINTS = ("바로", "지금", "빨리", "집중", "즉시", "당일", "먼저", "안되", "확인")
    BENIGN_HINTS = ("안녕", "고맙", "날씨", "점심", "주말", "가족", "회의")

    def segment(self, transcript: str, prosody: Optional[dict] = None) -> TIFE:
        txt = transcript or ""
        harm = sum(txt.count(h) for h in self.HARM_HINTS)
        press = sum(txt.count(p) for p in self.PRESSURE_HINTS)
        benign = sum(txt.count(b) for b in self.BENIGN_HINTS)

        # Independent signals — not normalized against each other (do not sum to 1).
        F = _sat(0.32 * harm + 0.08 * press)                    # harm>=2 -> >0.5
        T = _sat(0.50 * benign + (0.55 if harm == 0 and press == 0 else 0.0))  # clean text -> >0.75
        # Ignorance: high when the signal is weak (too little information) or conflicting (harm and benign coexist).
        weak = 1.0 if (harm + press + benign) == 0 else 0.0
        conflict = 1.0 if (harm > 0 and benign > 0) else 0.0
        I = _sat(0.70 * weak + 0.50 * conflict)                 # weak -> >0.65
        # Energy: prefer prosody when available; otherwise fall back to textual pressure hints.
        if prosody:
            E = _sat(prosody.get("energy", 0.0), soft=True)
        else:
            E = _sat(0.20 * press + 0.05 * harm)
        return TIFE(T, I, F, E).clamp()


def _sat(x: float, soft: bool = False) -> float:
    """Saturate to [0,1]. Default is a linear clamp (threshold crossing stays clear); soft=True is a soft saturation (for prosody already in [0,1])."""
    if x <= 0.0:
        return 0.0
    if soft:
        return min(1.0, x)
    return min(1.0, x)


def provider_from_callable(fn: Callable[[str, Optional[dict]], TIFE]) -> TIFEProvider:
    """Wrap an arbitrary callable as a TIFEProvider (for binding a real inference function)."""

    class _P:
        def segment(self, transcript: str, prosody: Optional[dict] = None) -> TIFE:
            return fn(transcript, prosody)

    return _P()


def stream_tife(provider: TIFEProvider, segments: List[dict]) -> List[TIFE]:
    """List of segments ({'transcript':..., 'prosody':...}) -> TIFE sequence."""
    out: List[TIFE] = []
    for seg in segments:
        out.append(provider.segment(seg.get("transcript", ""), seg.get("prosody")))
    return out
