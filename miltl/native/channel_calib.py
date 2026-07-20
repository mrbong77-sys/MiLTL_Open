"""Neutrosophic/affect channel membership functions + calibration (docs/ARCHITECTURE.md) — resolves zero-shot collapse.

**Design (docs/ARCHITECTURE.md)**: channels = neutrosophic SVNS membership functions (T/I/F independent, I = cross-modal contradiction), grounded in affect VAD.
Multiplication (AND collapse) → **soft-OR additive + calibrated logistic**. Activation threshold and sensitivity (gain) are defined via benign percentiles.

Pipeline:
  prosody[L,18] --(z-normalize, benign stats)--> z --(affect weighting)--> (A,V,D)
  (A,V,D)+speech-act+warmth --(soft-OR additive)--> evidence zF/zI/zT --(calibrated logistic)--> (T,I,F,E)
The calibrator is **fit on benign-train only** (frozen protocol, no KorMMP). Pure numpy (analytic teacher).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# Same order as nibble_features._PROS_KEYS (fixed index reference)
PROS_KEYS = ["voiced_ratio", "f0_mean", "f0_std", "f0_slope", "energy_mean", "energy_std",
             "energy_slope", "zcr_mean", "rate_proxy", "jitter", "shimmer", "hnr_mean",
             "spectral_centroid", "spectral_tilt", "f0_range", "pause_ratio", "pause_rate", "mean_pause_s"]
_IX = {k: i for i, k in enumerate(PROS_KEYS)}
_EPS = 1e-6

# ── Affect VAD weighting — docs/ARCHITECTURE.md. Linear combination of z-normalized features (design constants). ──
G_VAD = 1.5                                    # VAD logistic gain (median→0.5, ensures spread)


G_MAX = 10.0                                   # upper bound on channel logistic gain (prevents oversaturation = hard step)
_MIN_SPREAD = 0.05                             # lower bound on benign evidence spread (prevents gain blow-up on homogeneous samples)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def avd_from_z(z: np.ndarray) -> np.ndarray:
    """z[L,18] → (A,V,D)[L,3] ∈[0,1]. Linear combination of affect evidence + logistic. benign median≈0.5."""
    e, f0, rate = z[:, _IX["energy_mean"]], z[:, _IX["f0_mean"]], z[:, _IX["rate_proxy"]]
    frng, pause = z[:, _IX["f0_range"]], z[:, _IX["pause_ratio"]]
    tilt, hnr = z[:, _IX["spectral_tilt"]], z[:, _IX["hnr_mean"]]
    A = _sig(G_VAD * (0.4 * e + 0.4 * f0 + 0.2 * rate))
    D = _sig(G_VAD * (0.35 * e + 0.30 * frng - 0.20 * pause + 0.15 * rate))
    V = _sig(G_VAD * (-tilt + 0.3 * hnr))       # negative tilt = warm → high V
    return np.stack([A, V, D], axis=-1).astype(np.float32)


def evidence(avd: np.ndarray, sa: np.ndarray, warmth: np.ndarray, z_pause: np.ndarray) -> Dict[str, np.ndarray]:
    """(A,V,D)[L,3]+speech-act[L,4]+warmth[L] → per-nibble evidence zF/zI/zT (soft-OR additive, pre-logistic).

    speech-act axes: 0 directive · 1 urgency · 2 threat · 3 subversion.
    **Evidence components are selected by measured discriminative power (docs/ARCHITECTURE.md)** — no-signal/counterproductive components (D-dominance, hesitation) are excluded:
      D (AUROC 0.52) and hesitation carry no signal on real data yet have high variance → they swamp real signals (cold, subversion, XM).
    zF coercion = cold(0.72)+subversion(0.88)+threat(0.71)+directive  (D removed)
    zI latent = XM(0.69)+affinity×ask  (subversion moved to F, hesitation removed)
    zT natural = coherence+consistency+balanced D+prosocial−coercion (kept: AUROC 0.95).
    """
    A, V, D = avd[:, 0], avd[:, 1], avd[:, 2]
    directive, urgency, threat, subversion = sa[:, 0], sa[:, 1], sa[:, 2], sa[:, 3]
    cold = 1.0 - V
    # F (overt coercion): cold/harsh tone (real prosody signal) + scam ask + threat + directive. D removed (no-signal swamp).
    zF = 0.40 * cold + 0.30 * subversion + 0.20 * threat + 0.10 * directive
    # I (latent indeterminacy): cross-modal contradiction (warm lexicon + cold/dominant tone) + affinity×ask. hesitation removed (noise).
    XM = np.clip(warmth - V, 0.0, 1.0) * (0.5 + 0.5 * D)
    has_ask = np.maximum(directive, subversion)
    zI = 0.85 * XM + 0.15 * (warmth * has_ask)   # XM-centric (reduces warmth·ask contamination from financial consultations, docs/BENCHMARK.md)
    # T (natural, truth): coherence (call-level) is combined in channels/fit. Per-nibble components (AUROC 0.95, kept).
    balancedD = 1.0 - np.abs(D - 0.5) * 2.0
    coercion = np.maximum(threat, directive)
    zT_local = 0.30 * (1.0 - XM) + 0.20 * balancedD + 0.15 * warmth - 0.5 * coercion
    return {"A": A, "zF": zF, "zI": zI, "zT_local": zT_local, "XM": XM}


@dataclass
class Calib:
    """Calibration parameters (fit on benign-train). z-normalization stats + per-channel threshold/gain."""
    pros_med: np.ndarray = None            # [18]
    pros_iqr: np.ndarray = None            # [18]
    thr: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # {F,I,T:(b,g)}

    def zfeat(self, pros: np.ndarray) -> np.ndarray:
        """prosody[L,18] → z[L,18] (robust median/IQR, clip±3)."""
        return np.clip((pros - self.pros_med) / (self.pros_iqr + _EPS), -3.0, 3.0).astype(np.float32)

    def membership(self, key: str, zc: np.ndarray) -> np.ndarray:
        b, g = self.thr[key]
        return _sig(g * (zc - b)).astype(np.float32)

    def to_dict(self):
        return {"pros_med": self.pros_med.tolist(), "pros_iqr": self.pros_iqr.tolist(),
                "thr": {k: list(v) for k, v in self.thr.items()}}

    @classmethod
    def from_dict(cls, d):
        return cls(np.asarray(d["pros_med"], np.float32), np.asarray(d["pros_iqr"], np.float32),
                   {k: tuple(v) for k, v in d["thr"].items()})


def _thr_high(zc, hi=90, lo=10):
    """F/I: threshold at the benign upper (hi) percentile (most benign falls below). gain=4/spread, capped at G_MAX."""
    b = float(np.percentile(zc, hi))
    spread = max(float(np.percentile(zc, hi) - np.percentile(zc, lo)), _MIN_SPREAD)
    return b, min(4.0 / spread, G_MAX)


def _thr_low(zc, hi=80, lo=20):
    """T: threshold at the benign lower (lo) percentile (most benign lies above = high T)."""
    b = float(np.percentile(zc, lo))
    spread = max(float(np.percentile(zc, hi) - np.percentile(zc, lo)), _MIN_SPREAD)
    return b, min(4.0 / spread, G_MAX)


def fit_calib(benign_inputs) -> Calib:
    """List of benign NibbleChannelInput → Calib. Valid nibbles only. (docs/ARCHITECTURE.md calibration)"""
    pros = np.concatenate([nci.prosody[nci.mask > 0.5] for nci in benign_inputs], axis=0)
    med = np.median(pros, axis=0).astype(np.float32)
    iqr = (np.percentile(pros, 75, axis=0) - np.percentile(pros, 25, axis=0)).astype(np.float32)
    cal = Calib(med, iqr, {})
    # Compute channel evidence zF/zI/zT on benign → threshold/gain
    zF_all, zI_all, zT_all = [], [], []
    for nci in benign_inputs:
        m = nci.mask > 0.5
        if not m.any():
            continue
        z = cal.zfeat(nci.prosody)
        avd = avd_from_z(z)
        w = nci.warmth if nci.warmth is not None else np.zeros(len(z), np.float32)
        ev = evidence(avd, nci.speech_act, w, z[:, _IX["pause_ratio"]])
        coh = _coherence(ev["A"][m])
        zT = ev["zT_local"] + 0.35 * coh
        zF_all.append(ev["zF"][m]); zI_all.append(ev["zI"][m]); zT_all.append(zT[m])
    zF_all = np.concatenate(zF_all); zI_all = np.concatenate(zI_all); zT_all = np.concatenate(zT_all)
    cal.thr = {"F": _thr_high(zF_all), "I": _thr_high(zI_all), "T": _thr_low(zT_all)}
    return cal


def _coherence(a_valid: np.ndarray) -> float:
    """Call-level prosodic coherence: low arousal variability = high (entrainment proxy)."""
    if len(a_valid) < 2:
        return 0.5
    return float(np.clip(1.0 - np.std(a_valid), 0.0, 1.0))


def channels(nci, cal: Calib) -> np.ndarray:
    """NibbleChannelInput + Calib → tife[L,4] (T,I,F,E). Calibrated logistic membership functions."""
    L = nci.prosody.shape[0]
    out = np.zeros((L, 4), np.float32)
    m = nci.mask > 0.5
    if not m.any():
        return out
    z = cal.zfeat(nci.prosody)
    avd = avd_from_z(z)
    w = nci.warmth if nci.warmth is not None else np.zeros(L, np.float32)
    ev = evidence(avd, nci.speech_act, w, z[:, _IX["pause_ratio"]])
    coh = _coherence(ev["A"][m])
    zT = ev["zT_local"] + 0.35 * coh
    T = cal.membership("T", zT)
    I = cal.membership("I", ev["zI"])
    Fc = cal.membership("F", ev["zF"])
    E = ev["A"]                                  # E = arousal (already calibrated to [0,1])
    out[:, 0] = np.where(m, T, 0.0)
    out[:, 1] = np.where(m, I, 0.0)
    out[:, 2] = np.where(m, Fc, 0.0)
    out[:, 3] = np.where(m, E, 0.0)
    return out


def channels5(nci, cal: Calib) -> np.ndarray:
    """[L,5] = (T,I,F,E,XM). Input to the trainable head — makes XM (cross-modal contradiction = novelty) an explicit per-nibble channel (docs/BENCHMARK.md)."""
    L = nci.prosody.shape[0]
    out = np.zeros((L, 5), np.float32)
    out[:, :4] = channels(nci, cal)
    m = nci.mask > 0.5
    if m.any():
        z = cal.zfeat(nci.prosody)
        avd = avd_from_z(z)
        w = nci.warmth if nci.warmth is not None else np.zeros(L, np.float32)
        ev = evidence(avd, nci.speech_act, w, z[:, _IX["pause_ratio"]])
        out[:, 4] = np.where(m, ev["XM"], 0.0)
    return out


def _selftest() -> int:
    from dataclasses import dataclass as _dc
    rng = np.random.default_rng(0)

    @_dc
    class _NCI:
        prosody: np.ndarray; speech_act: np.ndarray; warmth: np.ndarray; mask: np.ndarray

    def mk(n, energy, f0, threat, directive, warmth, subversion=0.0):
        L = 26
        pros = np.zeros((L, 18), np.float32)
        pros[:n, _IX["energy_mean"]] = energy + 0.005 * rng.standard_normal(n)
        pros[:n, _IX["f0_mean"]] = f0 + 5 * rng.standard_normal(n)
        pros[:n, _IX["f0_range"]] = 40 + 30 * (energy > 0.08)
        pros[:n, _IX["spectral_tilt"]] = 1.0 if threat > 0.3 else -1.0
        sa = np.zeros((L, 4), np.float32)
        sa[:n, 0] = directive; sa[:n, 2] = threat; sa[:n, 3] = subversion
        wm = np.zeros(L, np.float32); wm[:n] = warmth
        mask = np.zeros(L, np.float32); mask[:n] = 1.0
        return _NCI(pros, sa, wm, mask)

    # 30 benign calls (friendly, low threat) + 30 fss calls (threat, directive, high energy)
    benign = [mk(20, 0.04, 180, 0.0, 0.1, 0.5) for _ in range(30)]
    fss = [mk(20, 0.12, 240, 0.7, 0.6, 0.1, subversion=0.6) for _ in range(30)]
    cal = fit_calib(benign)
    print(f"[selftest] Calib fit: thr F={tuple(round(x,3) for x in cal.thr['F'])} "
          f"I={tuple(round(x,3) for x in cal.thr['I'])} T={tuple(round(x,3) for x in cal.thr['T'])}")
    Fb = np.mean([channels(x, cal)[:20, 2].mean() for x in benign])
    Ff = np.mean([channels(x, cal)[:20, 2].mean() for x in fss])
    Tb = np.mean([channels(x, cal)[:20, 0].mean() for x in benign])
    Tf = np.mean([channels(x, cal)[:20, 0].mean() for x in fss])
    print(f"[selftest] F: benign={Fb:.3f}  fss={Ff:.3f}   (제로샷 탈출: fss F≫benign F)")
    print(f"[selftest] T: benign={Tb:.3f}  fss={Tf:.3f}   (benign T≫fss T)")
    assert Ff > Fb + 0.2, (Fb, Ff)
    assert Tb > Tf + 0.1, (Tb, Tf)
    assert Ff > 0.4, f"fss F 제로샷 탈출 실패: {Ff}"
    print("[selftest] 뉴트로소픽·정동 채널 캘리브레이션 동작 — 제로샷 탈출·soft-OR 분리 확인.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
