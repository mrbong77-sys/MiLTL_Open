"""Threshold recalibration — PEINN stays untouched (agnostic); only the thresholds that binarize
each channel's sigmoid output are reset on the real distribution (the docs/ARCHITECTURE.md
philosophy executed on real data).

Key observation (real FSS phishing vs AI-Hub benign): per-segment channels are weak (E is
strongest, AUC~0.71), but **per-channel firing rates accumulated over a 2-minute window separate
strongly** (E call-AUC~0.95, 4-channel combination ~0.96). The seed thresholds (v2.1 tau_F=0.5,
tau_I=0.65, etc.) kill the low F/I/E distributions of conversational text -> reset to the real
distribution boundary.

- calibrate_thresholds: search each channel threshold by **maximizing call-level separability
  (|AUC-0.5|)** (consistent with window-level decisions).
- re_encode_stream: apply thresholds to stored text_tife to re-derive text_nibble (no PEINN rerun).
Pure stdlib. Operates on CallStreams that carry TIFE values (default output of
build_streams_real, with_raw).
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import List, Optional

from .encoder import NibbleThresholds, NibbleEncoder

_CH = ("T", "I", "F", "E")


def _seg_tifes(cs, modality: str = "text"):
    """Per-segment TIFE of the given modality (missing excluded). modality: 'text'->text_tife, 'wave'->wave_tife."""
    attr = "wave_tife" if modality == "wave" else "text_tife"
    return [tf for s in cs.segments if (tf := getattr(s, attr)) is not None]


def _auc(pos: List[float], neg: List[float]) -> float:
    """Mann-Whitney U -> P(random pos > random neg). Channel/firing-rate separability (phishing = pos)."""
    if not pos or not neg:
        return 0.5
    negs = sorted(neg)
    s = 0.0
    for v in pos:
        lo = bisect.bisect_left(negs, v)
        hi = bisect.bisect_right(negs, v)
        s += lo + 0.5 * (hi - lo)
    return s / (len(pos) * len(neg))


def _call_rate(cs, ch: str, thr: float, modality: str = "text") -> Optional[float]:
    """Fraction of segments in the call with channel value >= threshold (2-minute cumulative firing rate)."""
    segs = _seg_tifes(cs, modality)
    if not segs:
        return None
    return sum(1 for tf in segs if getattr(tf, ch) >= thr) / len(segs)


@dataclass
class ChannelReport:
    seg_auc: float          # segment-level separability
    thr: float              # recalibrated threshold
    call_auc: float         # call-level firing-rate separability after applying the threshold (the key objective)
    phish_mean: float
    benign_mean: float


def calibrate_thresholds(phishing, benign, grid: int = 60, modality: str = "text"):
    """Phishing and benign CallStream lists -> per-channel recalibrated NibbleThresholds + report.

    Each channel threshold is searched by **maximizing call-level firing-rate |AUC-0.5|**
    (consistent with window-level separability). If I/F run in the reverse direction (higher on
    benign), AUC comes out < 0.5, but the separation magnitude is still valid — the downstream ML
    learns the sign. Use modality='wave' to apply symmetrically to the waveform channels
    (wave_tife) for native multimodality (see design notes)."""
    best_th, report = {}, {}
    for ch in _CH:
        pv = [getattr(tf, ch) for cs in phishing for tf in _seg_tifes(cs, modality)]
        bv = [getattr(tf, ch) for cs in benign for tf in _seg_tifes(cs, modality)]
        if not pv or not bv:
            best_th[ch] = 0.5
            continue
        seg_auc = _auc(pv, bv)
        pool = sorted(pv + bv)
        cands = sorted(set(pool[min(len(pool) - 1, int(i * (len(pool) - 1) / grid))]
                           for i in range(1, grid)))
        best = (0.0, pool[len(pool) // 2], 0.5)      # (|auc-.5|, thr, call_auc)
        for t in cands:
            pr = [r for cs in phishing if (r := _call_rate(cs, ch, t, modality)) is not None]
            br = [r for cs in benign if (r := _call_rate(cs, ch, t, modality)) is not None]
            ca = _auc(pr, br)
            if abs(ca - 0.5) > best[0]:
                best = (abs(ca - 0.5), t, ca)
        best_th[ch] = round(best[1], 4)
        report[ch] = ChannelReport(round(seg_auc, 3), round(best[1], 4), round(best[2], 3),
                                   round(sum(pv) / len(pv), 3), round(sum(bv) / len(bv), 3))
    th = NibbleThresholds(tau_T=best_th["T"], tau_I=best_th["I"], tau_F=best_th["F"],
                          tau_E=best_th["E"], tau_C=0.30, use_conflict=False)
    return th, report


def _call_feats(cs, th: NibbleThresholds, modality: str = "text"):
    """Call -> 4-channel call firing-rate vector [T,I,F,E]. None if the call has no segments."""
    fr = [_call_rate(cs, ch, getattr(th, f"tau_{ch}"), modality) for ch in _CH]
    return fr if all(x is not None for x in fr) else None


def _call_feats_fused(cs, text_th: NibbleThresholds, wave_th: NibbleThresholds):
    """Call -> 8-channel fused firing rates [T,I,F,E(text) | T,I,F,E(wave)] (native multimodal).

    Valid only when both modalities have segments — None if either is missing (not a fusion candidate)."""
    ft = _call_feats(cs, text_th, "text")
    fw = _call_feats(cs, wave_th, "wave")
    return (ft + fw) if (ft and fw) else None


def train_combined_gate(phishing, benign, th: NibbleThresholds, modality: str = "text"):
    """Phishing vs benign call firing rates -> logistic (combined decider). For held-out evaluation."""
    from .gate1_train import LogisticRegression
    X, y = [], []
    for cs in phishing:
        f = _call_feats(cs, th, modality)
        if f:
            X.append(f); y.append(1)
    for cs in benign:
        f = _call_feats(cs, th, modality)
        if f:
            X.append(f); y.append(0)
    return LogisticRegression(epochs=800, lr=0.3).fit(X, y)


def score_call(cs, th: NibbleThresholds, model, modality: str = "text") -> Optional[float]:
    """Call phishing score in [0,1] (combined decider)."""
    f = _call_feats(cs, th, modality)
    return model.predict_proba(f) if f else None


def combined_call_auc(phishing, benign, th: NibbleThresholds, modality: str = "text"):
    """4-channel call firing rates -> logistic -> call-level combined separability (in-sample, for metrics)."""
    m = train_combined_gate(phishing, benign, th, modality)
    sp = [s for cs in phishing if (s := score_call(cs, th, m, modality)) is not None]
    sb = [s for cs in benign if (s := score_call(cs, th, m, modality)) is not None]
    return _auc(sp, sb), list(m.w)


def fused_call_auc(phishing, benign, text_th: NibbleThresholds, wave_th: NibbleThresholds):
    """8-channel fused firing rates (text and wave) -> logistic -> call-level fused separability.

    Only calls carrying both modalities are used (upper-bound metric of native multimodal fusion).
    Returns: (fused_auc, text_only_auc, wave_only_auc, n_phish, n_benign) — the three deciders are
    compared on the same call set to quantify the fusion gain."""
    from .gate1_train import LogisticRegression
    Xf, Xt, Xw, y = [], [], [], []
    for cs, lab in [(c, 1) for c in phishing] + [(c, 0) for c in benign]:
        ft = _call_feats(cs, text_th, "text")
        fw = _call_feats(cs, wave_th, "wave")
        if ft and fw:                                # fusion requires both modalities
            Xf.append(ft + fw); Xt.append(ft); Xw.append(fw); y.append(lab)
    np_, nn = sum(y), len(y) - sum(y)
    if np_ == 0 or nn == 0:
        return None, None, None, np_, nn

    def _auc_of(X):
        m = LogisticRegression(epochs=800, lr=0.3).fit(X, y)
        sp = [m.predict_proba(x) for x, l in zip(X, y) if l == 1]
        sb = [m.predict_proba(x) for x, l in zip(X, y) if l == 0]
        return _auc(sp, sb)

    return _auc_of(Xf), _auc_of(Xt), _auc_of(Xw), np_, nn


def pair_benign_modalities(text_calls, wave_calls, limit: int = 0):
    """Workaround for the lack of dual-modal benign data — pair independent benign text and wave calls into synthetic dual-modal benign calls.

    Pairs one benign text call (text_tife) with one benign waveform call (wave_tife). Since both
    are benign, the joint is benign too (label-preserving). Phishing uses real pairs (fss_audio);
    only benign is filled by this augmentation, enabling fused call-AUC measurement. Assumption:
    the two benign channels are independent (ignoring within-benign correlation = conservative
    estimate). Aligned by segment idx (min length). Each channel is taken verbatim from its
    original call — no re-featurization (PEINN untouched)."""
    from .schema import CallStream, SegmentRecord
    tcs = [c for c in text_calls if _seg_tifes(c, "text")]
    wcs = [c for c in wave_calls if _seg_tifes(c, "wave")]
    n = min(len(tcs), len(wcs))
    if limit:
        n = min(n, limit)
    out = []
    for i in range(n):
        tsegs = [s for s in tcs[i].segments if s.text_tife is not None]
        wsegs = [s for s in wcs[i].segments if s.wave_tife is not None]
        m = min(len(tsegs), len(wsegs))
        segs = [SegmentRecord(idx=j, t0=j * 8.0, t1=(j + 1) * 8.0,
                              text_nibble=None, wave_nibble=None,
                              text_tife=tsegs[j].text_tife, wave_tife=wsegs[j].wave_tife)
                for j in range(m)]
        out.append(CallStream(call_id=f"pair_{i:04d}", source="benign_paired",
                              label="benign", segments=segs, segment_seconds=8.0))
    return out


def re_encode_stream(cs, text_th: NibbleThresholds, wave_th: Optional[NibbleThresholds] = None):
    """Apply thresholds to stored text_tife/wave_tife to re-derive nibbles (recalibration without rerunning PEINN). In-place."""
    enc_t = NibbleEncoder(text_th)
    enc_w = NibbleEncoder(wave_th) if wave_th else None
    for s in cs.segments:
        if s.text_tife is not None:
            s.text_nibble = enc_t.encode(s.text_tife)
        if enc_w and s.wave_tife is not None:
            s.wave_nibble = enc_w.encode(s.wave_tife)
    return cs
