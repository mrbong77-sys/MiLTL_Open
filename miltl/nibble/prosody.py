"""Prosody extractor — raw PCM waveform → per-segment acoustic features (see design notes).

Front-end of the waveform-nibble path. For each 8-second segment it extracts pitch (F0),
energy, speaking rate, jitter, etc. as input to wave_featurize (the acoustic T/I/F/E head).
numpy-based (audio is numerically intensive).

⚠️ Only this module requires numpy (the package __init__ does not import it → the text path
needs no numpy). On DGX/edge it can be swapped for a faster backend (librosa MFCC, etc.)
behind the same interface.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np

# F0 search range (human voice): 75-300 Hz
F0_MIN, F0_MAX = 75.0, 300.0


def read_pcm(data: bytes, bits: int = 16, channels: int = 1) -> np.ndarray:
    """raw PCM bytes → float32 [-1,1] mono samples. (headerless pcm; caller manages sr)

    Real data (KsponSpeech etc.) can contain odd/trailing bytes, so we truncate to a
    multiple of the sample width (× channels) — otherwise np.frombuffer raises
    'buffer size must be a multiple of element size', and the caller's except swallows
    it into a silent zero-call."""
    dt = {8: np.int8, 16: "<i2", 32: "<i4"}[bits]
    isz = bits // 8  # frombuffer requires the byte count to be a multiple of itemsize
    if isz > 1 and (len(data) % isz):
        data = data[: len(data) - (len(data) % isz)]
    x = np.frombuffer(data, dtype=dt).astype(np.float32)
    if bits == 8:
        x = (x - 128.0) / 128.0
    else:
        x = x / float(1 << (bits - 1))
    if channels > 1:
        x = x[: len(x) // channels * channels].reshape(-1, channels).mean(axis=1)
    return x


def _frames(x: np.ndarray, sr: int, win_ms: float = 25.0, hop_ms: float = 10.0) -> np.ndarray:
    win = max(1, int(sr * win_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    if len(x) < win:
        return np.empty((0, win), dtype=np.float32)
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def _f0_autocorr(frame: np.ndarray, sr: int):
    """Per-frame (F0 Hz, ac_ratio). (0, 0) if unvoiced. ac_ratio = normalized autocorrelation peak (HNR source)."""
    f = frame - frame.mean()
    e = float(np.dot(f, f))
    if e < 1e-6:
        return 0.0, 0.0
    r = np.correlate(f, f, mode="full")[len(f) - 1:]
    lo, hi = int(sr / F0_MAX), int(sr / F0_MIN)
    if hi <= lo or hi >= len(r):
        return 0.0, 0.0
    seg = r[lo:hi]
    lag = lo + int(np.argmax(seg))
    ratio = float(r[lag] / r[0]) if r[0] > 0 else 0.0
    if ratio < 0.3:                              # weak periodicity → unvoiced
        return 0.0, 0.0
    return sr / lag, ratio


@dataclass
class ProsodyFeatures:
    """One segment's acoustic features (raw, pre-normalization). AVD grounding (docs/BASELINES.md): A=arousal, V=valence, D=dominance."""
    dur_s: float
    voiced_ratio: float          # voiced frame ratio (speech density)
    f0_mean: float               # mean pitch (Hz, voiced)             [A/D]
    f0_std: float                # pitch variability                   [D]
    f0_slope: float              # pitch rising trend (Hz/s; positive=urgent)  [A trajectory]
    energy_mean: float           # RMS                          [A]
    energy_std: float
    energy_slope: float          # energy rising trend (/s)     [A trajectory]
    zcr_mean: float              # zero-crossing rate (noise/fricatives)
    rate_proxy: float            # voiced→unvoiced transitions per second (syllable rate)  [A]
    # ── docs/BASELINES.md extensions ──
    jitter: float = 0.0          # local F0 variability (|Δf0|/f0, voiced)   [F/stress]
    shimmer: float = 0.0         # local amplitude variability (|Δrms|/rms)  [F/stress]
    hnr_mean: float = 0.0        # harmonics-to-noise ratio (dB, voiced clarity)  [V: low=harsh/rough]
    spectral_centroid: float = 0.0   # spectral centroid (Hz, brightness) [A/harshness]
    spectral_tilt: float = 0.0   # log-spectrum slope (negative=low-band=warm) [V]
    f0_range: float = 0.0        # F0 dynamic range (P90-P10, intonation span)  [D/assertiveness]
    pause_ratio: float = 0.0     # low-energy (silent) frame ratio    [I/hesitation]
    pause_rate: float = 0.0      # silent runs per second             [I/rhythm]
    mean_pause_s: float = 0.0    # mean silence length (s)            [I/hesitation]

    def as_dict(self) -> dict:
        return asdict(self)


def _runs_of(mask: np.ndarray) -> int:
    """Number of True runs (contiguous spans)."""
    if mask.size == 0:
        return 0
    return int(mask[0]) + int(np.sum((mask[1:].astype(int) - mask[:-1].astype(int)) == 1))


def segment_prosody(x: np.ndarray, sr: int, hop_s: float = 0.01) -> ProsodyFeatures:
    """Segment waveform → ProsodyFeatures (including AVD-grounded features, docs/BASELINES.md)."""
    fr = _frames(x, sr)
    dur = len(x) / sr if sr else 0.0
    if len(fr) == 0:
        return ProsodyFeatures(dur, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    rms = np.sqrt((fr ** 2).mean(axis=1) + 1e-12)
    zcr = (np.abs(np.diff(np.sign(fr), axis=1)) > 0).mean(axis=1)
    fa = [_f0_autocorr(fr[i], sr) for i in range(len(fr))]
    f0 = np.array([a[0] for a in fa]); acr = np.array([a[1] for a in fa])
    voiced = f0 > 0
    vr = float(voiced.mean())
    t = np.arange(len(fr)) * hop_s
    f0v = f0[voiced]
    if f0v.size >= 2:
        f0_mean, f0_std = float(f0v.mean()), float(f0v.std())
        f0_slope = float(np.polyfit(t[voiced], f0v, 1)[0])
        f0_range = float(np.percentile(f0v, 90) - np.percentile(f0v, 10))   # [D]
        jitter = float(np.abs(np.diff(f0v)).mean() / (f0v.mean() + 1e-9))    # [F/stress]
    elif f0v.size == 1:
        f0_mean, f0_std, f0_slope, f0_range, jitter = float(f0v[0]), 0.0, 0.0, 0.0, 0.0
    else:
        f0_mean = f0_std = f0_slope = f0_range = jitter = 0.0
    en_slope = float(np.polyfit(t, rms, 1)[0]) if len(fr) >= 2 else 0.0
    transitions = int(np.sum((voiced[1:].astype(int) - voiced[:-1].astype(int)) == 1))
    rate = transitions / dur if dur > 0 else 0.0

    # shimmer: local amplitude variability of voiced frames [F/stress]
    rv = rms[voiced]
    shimmer = float(np.abs(np.diff(rv)).mean() / (rv.mean() + 1e-9)) if rv.size >= 2 else 0.0
    # HNR: voiced autocorrelation ratio → dB (clarity, low=harsh) [V]
    av = acr[voiced]; av = np.clip(av, 1e-4, 0.9999)
    hnr = float((10.0 * np.log10(av / (1.0 - av))).mean()) if av.size else 0.0
    # spectrum: mean magnitude spectrum → centroid [A/brightness], tilt [V, negative=low-band=warm]
    mag = np.abs(np.fft.rfft(fr * np.hanning(fr.shape[1]), axis=1)).mean(0) + 1e-9
    freqs = np.fft.rfftfreq(fr.shape[1], d=1.0 / sr)
    centroid = float((freqs * mag).sum() / mag.sum())
    tilt = float(np.polyfit(freqs, np.log(mag), 1)[0] * 1000.0)              # log-mag/kHz
    # pause: low-energy silent frames [I/hesitation]
    silent = rms < 0.2 * (np.median(rms) + 1e-9)
    pause_ratio = float(silent.mean())
    pcount = _runs_of(silent)
    pause_rate = pcount / dur if dur > 0 else 0.0
    mean_pause_s = float(silent.sum() * hop_s / pcount) if pcount else 0.0

    return ProsodyFeatures(dur, vr, f0_mean, f0_std, f0_slope,
                           float(rms.mean()), float(rms.std()), en_slope,
                           float(zcr.mean()), rate,
                           jitter, shimmer, hnr, centroid, tilt, f0_range,
                           pause_ratio, pause_rate, mean_pause_s)


def prosody_stream(x: np.ndarray, sr: int, seconds_per_seg: float = 8.0) -> List[ProsodyFeatures]:
    """Tile the call waveform into non-overlapping 8-second segments → per-segment ProsodyFeatures sequence (aligned with text tiling)."""
    seg = max(1, int(sr * seconds_per_seg))
    return [segment_prosody(x[i:i + seg], sr) for i in range(0, len(x), seg) if (len(x) - i) > sr * 0.3]
