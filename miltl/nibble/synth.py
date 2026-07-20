"""Synthetic PEINN output stream generator — for threshold simulation.

Note: this is NOT building a new nibble corpus. Using PEINN as-is is the final form; here we
cannot run the real PEINN inside the container, so we **mimic its continuous output (T,I,F,E)
distribution and temporal structure**. On DGX, replace `synth_call(...)` with real PEINN
per-segment inference (neutro head T/I/F + prosody-energy E) and the simulation harness
(simulate.py) runs unchanged on real streams.

Mimicry principles (reflecting FSS real-call dynamics):
  benign call   : high T, low F, low E, intermittent ambiguity (I).
  phishing call : **emotion/intent rising over time** — early ambiguity (I↑, F↓), mid-to-late
                  joint rise of F and E (harm intent + urgency/pressure co-firing). Per-segment
                  noise overlaps with benign.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

from .tife import TIFE


@dataclass
class SynthCall:
    """Continuous PEINN output sequence for one call + call label."""
    tife: List[TIFE]           # per-segment (T,I,F,E)
    label: int                 # 0=benign, 1=phishing

    def __len__(self) -> int:
        return len(self.tife)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _g(rng: random.Random, mean: float, sd: float) -> float:
    return _clamp01(rng.gauss(mean, sd))


def synth_call(rng: random.Random, label: int, n_seg: int) -> SynthCall:
    """Generate one call's (T,I,F,E) stream. label=0 benign / 1 phishing."""
    # Per-call individual variation (speaker/channel): even in phishing, a "polite scammer"
    # applies weaker pressure → within-class variation.
    polite = rng.random() < 0.35 if label == 1 else False
    segs: List[TIFE] = []
    for t in range(n_seg):
        p = t / max(1, n_seg - 1)          # progress within the call [0,1]
        r = rng.random()
        if label == 0:
            if r < 0.15:                    # benign but money/contract talk → slight F rise (causes overlap)
                T, I, F, E = _g(rng, 0.50, 0.20), _g(rng, 0.25, 0.15), _g(rng, 0.38, 0.16), _g(rng, 0.22, 0.14)
            elif r < 0.27:                  # ambiguous (greetings/small talk)
                T, I, F, E = _g(rng, 0.45, 0.20), _g(rng, 0.55, 0.20), _g(rng, 0.12, 0.10), _g(rng, 0.15, 0.12)
            else:                           # typical benign
                T, I, F, E = _g(rng, 0.62, 0.20), _g(rng, 0.22, 0.15), _g(rng, 0.12, 0.10), _g(rng, 0.15, 0.12)
        else:
            # Phishing: early ambiguity (impersonation/rapport) → mid-to-late rise in harm and urgency.
            # Large noise (segments overlap with benign).
            damp = 0.55 if polite else 1.0
            F = _g(rng, (0.15 + 0.45 * p) * damp, 0.20)
            E = _g(rng, (0.18 + 0.45 * p) * damp, 0.20)
            I = _g(rng, 0.20 + 0.40 * (1.0 - p), 0.17)   # high early, decaying
            T = _g(rng, 0.55 - 0.25 * p, 0.18)           # impersonation looks normal early, then declines
        segs.append(TIFE(T, I, F, E))
    return SynthCall(segs, label)


def synth_dataset(
    n_per_class: int = 200,
    seg_range: Tuple[int, int] = (24, 60),   # ~1min(24) to ~2.5min(60), segment≈2.5s
    seed: int = 20260702,
) -> List[SynthCall]:
    """Generate n_per_class calls each for benign/phishing (reproducible). Segment count random per call."""
    rng = random.Random(seed)
    calls: List[SynthCall] = []
    for label in (0, 1):
        for _ in range(n_per_class):
            n_seg = rng.randint(seg_range[0], seg_range[1])
            calls.append(synth_call(rng, label, n_seg))
    rng.shuffle(calls)
    return calls


def synth_mm_dataset(
    n_per_class: int = 200,
    seg_range: Tuple[int, int] = (12, 40),   # ~1.5–5 min at 8-second segments
    text_only_frac: float = 0.25,            # some calls lack audio (text-only) — mimics real mixture
    seed: int = 20260703,
):
    """Generate a list of multimodal synthetic CallStreams — for Gate-1/contract verification.

    Text and audio are generated with the **same label, independent noise** (the two modalities
    carry the label complementarily → multimodal fusion should beat unimodal). Some calls lack
    audio (verifies mask handling).
    build_call_stream produces the CallStream (contract).
    """
    from .schema import build_call_stream
    from .encoder import NibbleEncoder, NibbleThresholds
    rng = random.Random(seed)
    enc = NibbleEncoder(NibbleThresholds())
    calls = []
    for label in (0, 1):
        for i in range(n_per_class):
            n_seg = rng.randint(seg_range[0], seg_range[1])
            text = synth_call(rng, label, n_seg).tife
            wave = synth_call(rng, label, n_seg).tife            # independent noise, same label
            wave_stream = None if rng.random() < text_only_frac else wave
            calls.append(build_call_stream(
                call_id=f"{'phish' if label else 'benign'}_{i}",
                source="synthetic", label=("phishing" if label else "benign"),
                text_tifes=text, wave_tifes=wave_stream, text_enc=enc, wave_enc=enc,
                seconds_per_seg=8.0))
    rng.shuffle(calls)
    return calls
