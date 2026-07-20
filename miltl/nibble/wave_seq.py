"""wave prosody sequence → (L,C) matrix — for the learned wave Seq classifier (docs/BASELINES.md, approach B).

Approach B (user-confirmed): instead of a wave→T/I/F/E distill head (teacher is hard to
obtain), a CNN **learns the raw prosody feature sequence directly against harm/benign
labels**. Isomorphic to the text MiLTL-Seq (supervised waveform patterns), no teacher needed.

per-nibble (8s) prosody (16, wave_head.FEATURES) → +trajectory (diff·cum, with_trajectory) → zscore → (L,C=48).
Train/infer via `MultiScaleCNNAdaptor.fit_matrices/score_matrix`. Late-fusion (wave contributes only for calls with audio).

Calls without audio (korccvid·callcenter) leave wave inactive → text fallback (fusion gating).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .wave_head import FEATURES, feat_vec, with_trajectory


def prosody_vecs(prosody_list: Sequence) -> np.ndarray:
    """List of ProsodyFeatures (or dicts) → (n,16) raw features. (pre-normalization)"""
    def _d(p):
        return p.as_dict() if hasattr(p, "as_dict") else dict(p)
    if not prosody_list:
        return np.zeros((0, len(FEATURES)), np.float32)
    return np.stack([feat_vec(_d(p)) for p in prosody_list]).astype(np.float32)


def fit_norm(seqs: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """All-call (n,16) features → per-channel mean/std (for zscore). Computed once on the train set, shipped with the checkpoint."""
    allv = np.vstack([s for s in seqs if len(s)]) if any(len(s) for s in seqs) else np.zeros((1, len(FEATURES)))
    mean = allv.mean(axis=0).astype(np.float32)
    std = (allv.std(axis=0) + 1e-6).astype(np.float32)
    return mean, std


def seq_matrix(vecs: np.ndarray, mean: np.ndarray, std: np.ndarray, L: int = 26) -> np.ndarray:
    """(n,16) raw prosody → (L, C) matrix. zscore → +trajectory (diff·cum) → first L (observation budget), front zero-pad + mask.

    C = 16*3 (seg·diff·cum) + 1 (mask) = 49. Pressure lives in the trajectory (rising trend)
    (docs/BASELINES.md) → diff/cum are essential."""
    d = len(FEATURES)
    if len(vecs) == 0:
        return np.zeros((L, 3 * d + 1), np.float32)
    z = (vecs[:L] - mean) / std                              # zscore (mitigates saturation/scale)
    T = with_trajectory(z.astype(np.float32))                # (n, 3d) seg⊕diff⊕cum
    n = len(T); off = L - n
    M = np.zeros((L, 3 * d + 1), np.float32)
    if n >= L:
        M[:, :3 * d] = T[-L:]; M[:, -1] = 1.0                # beyond L keep first L (observation budget)… in practice off≤0
    else:
        M[off:, :3 * d] = T; M[off:, -1] = 1.0               # if short, front zero-pad + mask
    return M


# ── Audio → per-nibble prosody sequence (DGX; decode + prosody) ──────────────────
def audio_to_prosody(audio_uri, sr: int = 16000, seconds_per_seg: float = 8.0,
                     budget_segs: int = 26, telephone: bool = True, codec_equalize: bool = False) -> List:
    """audio_uri (str or **list**) → prosody for the first budget_segs 8-second nibbles. Capped at the observation budget (≈200s).

    ★list = multiple pcm files (e.g. ksponspeech_dual utterance pieces) → **decode then concatenate** into one call waveform (binary classification).
    codec_equalize=True (docs/BENCHMARK.md): uniform telephone-band + μ-law codec equalization — removes corpus codec leakage.
    """
    from .audio_decode import decode_to_pcm, telephone_band, equalize_channel
    from .prosody import prosody_stream
    uris = list(audio_uri) if isinstance(audio_uri, (list, tuple)) else [audio_uri]
    parts, got = [], sr
    for u in uris:
        try:
            xi, gi = decode_to_pcm(u, sr=sr)
        except Exception:  # noqa: BLE001
            continue
        if xi is not None and len(xi):
            parts.append(xi); got = gi or sr
    if not parts:
        return []
    x = np.concatenate(parts) if len(parts) > 1 else parts[0]
    if codec_equalize:
        x = equalize_channel(x, got, codec=True)            # telephone band + μ-law (fair equalization)
    elif telephone:
        x = telephone_band(x, got)                          # telephone band (300–3400) — matches real calls
    segs = prosody_stream(x, got, seconds_per_seg=seconds_per_seg)
    return segs[:budget_segs]
