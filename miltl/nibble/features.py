"""Nibble sequence → pattern feature vector (Gate-1 input). Pure stdlib.

Features (docs/ARCHITECTURE.md): 16-state histogram, F/E persistence, F∧E co-firing, I load,
time-to-first-harm, harm ramp, transition bigram top-k.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .encoder import unpack


@dataclass
class NibbleFeatures:
    n_segments: int
    state_hist: List[float]                 # 16-dim normalized frequencies
    f_rate: float                           # fraction with bF=1
    f_max_run: float                        # max F run-length (normalized by segment count)
    e_rate: float                           # fraction with bE=1
    fe_cooccur: float                       # fraction with bF∧bE (harm + pressure together)
    i_load: float                           # fraction with bI=1
    t_rate: float                           # fraction with bT=1
    time_to_first_harm: float               # index of first bF=1 / n (1.0 if none)
    harm_ramp: float                        # cumulative-F slope (linear approx, normalized to [0,1])
    top_transitions: List[Tuple[int, int, int]] = field(default_factory=list)  # (from,to,count)

    def to_vector(self) -> List[float]:
        """Flat feature vector (scorer input)."""
        return [
            self.f_rate, self.f_max_run, self.e_rate, self.fe_cooccur,
            self.i_load, self.t_rate, self.time_to_first_harm, self.harm_ramp,
        ] + self.state_hist


def _max_run(bits: List[int]) -> int:
    best = cur = 0
    for b in bits:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return best


def extract_features(nibbles: List[int], top_k: int = 5) -> NibbleFeatures:
    n = len(nibbles)
    hist = [0.0] * 16
    if n == 0:
        return NibbleFeatures(0, hist, 0, 0, 0, 0, 0, 0, 1.0, 0.0, [])

    bT = bI = bF = bE = []
    Ts, Is, Fs, Es = [], [], [], []
    for nib in nibbles:
        hist[nib] += 1.0
        t, i, f, e = unpack(nib)
        Ts.append(t); Is.append(i); Fs.append(f); Es.append(e)
    hist = [c / n for c in hist]

    f_rate = sum(Fs) / n
    e_rate = sum(Es) / n
    i_load = sum(Is) / n
    t_rate = sum(Ts) / n
    fe_cooccur = sum(1 for f, e in zip(Fs, Es) if f and e) / n
    f_max_run = _max_run(Fs) / n

    # time-to-first-harm
    ttf = next((idx for idx, f in enumerate(Fs) if f), None)
    time_to_first_harm = 1.0 if ttf is None else ttf / n

    # harm ramp: difference in F rate, first half vs second half → positive if rising (piecewise, clamped to [0,1])
    half = max(1, n // 2)
    early = sum(Fs[:half]) / half
    late = sum(Fs[half:]) / max(1, n - half)
    harm_ramp = max(0.0, min(1.0, (late - early + 1.0) / 2.0))

    # transition bigram top-k
    trans: Dict[Tuple[int, int], int] = {}
    for a, b in zip(nibbles, nibbles[1:]):
        trans[(a, b)] = trans.get((a, b), 0) + 1
    top = sorted(trans.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    top_transitions = [(a, b, c) for (a, b), c in top]

    return NibbleFeatures(
        n_segments=n, state_hist=hist, f_rate=f_rate, f_max_run=f_max_run,
        e_rate=e_rate, fe_cooccur=fe_cooccur, i_load=i_load, t_rate=t_rate,
        time_to_first_harm=time_to_first_harm, harm_ramp=harm_ramp,
        top_transitions=top_transitions,
    )
