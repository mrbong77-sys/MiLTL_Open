"""Multimodal nibble features + Gate-1 (docs/ARCHITECTURE.md) — consumes CallStream directly.

From a window's text nibble sequence + acoustic nibble sequence (missing entries allowed as None), extract:
  - per-modality pattern features (reusing features.extract_features),
  - **cross-modal features** (both modalities F, modal disagreement, etc. — docs/ARCHITECTURE.md, "modal disagreement is a signal"),
  - modality presence rates (masks).
From these, Gate-1 (an ultra-lightweight ML stand-in rule) produces p1=P(phishing). Missing modalities are handled via masks, not treated as 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from .features import NibbleFeatures, extract_features
from .encoder import unpack


@dataclass
class MMFeatures:
    n: int                                  # number of segments in the window
    text: Optional[NibbleFeatures]          # text nibble pattern (None if absent)
    wave: Optional[NibbleFeatures]          # acoustic nibble pattern (None if absent)
    text_present: float                     # fraction of segments with text
    wave_present: float                     # fraction of segments with audio
    both_present: float                     # fraction with both modalities present
    both_f: float                           # both modalities F=1 (strong multimodal phishing)
    text_f_only: float                      # text F but audio ¬F (script-reading type: risky content, calm tone)
    wave_f_only: float                      # audio F but text ¬F (early social engineering: pressuring tone, benign content)
    both_e: float                           # both modalities E=1 (simultaneous urgency)
    # Raw window sequences (consumed directly by a sequential Gate-1 model — aggregate features and sequential models both use the same MMFeatures)
    text_nibbles: List[Optional[int]] = field(default_factory=list)
    wave_nibbles: List[Optional[int]] = field(default_factory=list)

    @property
    def modal_disagree(self) -> float:
        return self.text_f_only + self.wave_f_only


# Fixed feature vector (fixed order) consumed by the learned Gate-1. Missing modalities are encoded as 0 + a present flag.
FEATURE_NAMES = (
    "t_f_rate", "t_f_max_run", "t_e_rate", "t_fe_cooccur", "t_i_load", "t_t_rate",
    "t_ttf_harm", "t_harm_ramp", "text_present",
    "w_f_rate", "w_f_max_run", "w_e_rate", "w_fe_cooccur", "w_i_load", "w_t_rate",
    "w_ttf_harm", "w_harm_ramp", "wave_present",
    "both_f", "text_f_only", "wave_f_only", "both_e", "both_present",
)


def _nf_vec(nf: Optional[NibbleFeatures]):
    if nf is None or nf.n_segments == 0:
        return [0.0] * 8
    return [nf.f_rate, nf.f_max_run, nf.e_rate, nf.fe_cooccur,
            nf.i_load, nf.t_rate, nf.time_to_first_harm, nf.harm_ramp]


def mm_feature_vector(mm: MMFeatures) -> List[float]:
    """MMFeatures → fixed-length float vector in FEATURE_NAMES order. Shared by rule-based and learned Gate-1."""
    return (_nf_vec(mm.text) + [mm.text_present]
            + _nf_vec(mm.wave) + [mm.wave_present]
            + [mm.both_f, mm.text_f_only, mm.wave_f_only, mm.both_e, mm.both_present])


def _harm_energy_bits(nib: int):
    _t, _i, f, e = unpack(nib)
    return f, e


def extract_mm_features(text_nibbles: List[Optional[int]],
                        wave_nibbles: List[Optional[int]]) -> MMFeatures:
    """Two modality nibble sequences of a window → MMFeatures. Assumes equal lengths (segment-aligned)."""
    n = max(len(text_nibbles), len(wave_nibbles))
    t_nz = [x for x in text_nibbles if x is not None]
    w_nz = [x for x in wave_nibbles if x is not None]
    text = extract_features(t_nz) if t_nz else None
    wave = extract_features(w_nz) if w_nz else None

    both = both_f = tf_only = wf_only = both_e = 0
    for tn, wn in zip(text_nibbles, wave_nibbles):
        if tn is None or wn is None:
            continue
        both += 1
        tf, te = _harm_energy_bits(tn)
        wf, we = _harm_energy_bits(wn)
        both_f += 1 if (tf and wf) else 0
        tf_only += 1 if (tf and not wf) else 0
        wf_only += 1 if (wf and not tf) else 0
        both_e += 1 if (te and we) else 0

    d = max(1, n)
    return MMFeatures(
        n=n, text=text, wave=wave,
        text_nibbles=list(text_nibbles), wave_nibbles=list(wave_nibbles),
        text_present=len(t_nz) / d, wave_present=len(w_nz) / d, both_present=both / d,
        both_f=both_f / d, text_f_only=tf_only / d, wave_f_only=wf_only / d, both_e=both_e / d,
    )


@dataclass
class MultimodalGate1:
    """Multimodal rule + logistic Gate-1 (cold start). Later swappable for a learned model (1D-CNN/GRU) behind the same interface.

    Missing modalities are weighted by their presence rate → a text-only call is naturally judged from text alone.
    Cross-modal features (both_f as a strong signal, disagreement) catch what a single modality misses.
    """
    # Weights shared across modalities (text and audio have identical structure, so applied symmetrically)
    w_f_rate: float = 1.8
    w_fe_cooccur: float = 2.2
    w_harm_ramp: float = 1.0
    w_t_rate: float = -1.6
    w_i_load: float = -0.3
    # Cross-modal
    w_both_f: float = 2.8          # both modalities F simultaneously → strongest phishing evidence
    w_disagree: float = 0.15       # modal disagreement → weak suspicion. A real-world signal, but needs calibration via training
    bias: float = -1.3

    def _modal_logit(self, nf: Optional[NibbleFeatures], present: float) -> float:
        if nf is None or nf.n_segments == 0:
            return 0.0
        s = (self.w_f_rate * nf.f_rate + self.w_fe_cooccur * nf.fe_cooccur
             + self.w_harm_ramp * nf.harm_ramp + self.w_t_rate * nf.t_rate
             + self.w_i_load * nf.i_load)
        return present * s

    def logit(self, mm: MMFeatures) -> float:
        return (self.bias
                + self._modal_logit(mm.text, mm.text_present)
                + self._modal_logit(mm.wave, mm.wave_present)
                + self.w_both_f * mm.both_f
                + self.w_disagree * mm.modal_disagree)

    def score(self, mm: MMFeatures) -> float:
        if mm.n == 0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-self.logit(mm)))
