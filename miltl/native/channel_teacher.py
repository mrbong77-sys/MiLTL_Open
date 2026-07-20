"""Channel teacher synthesis (docs/ARCHITECTURE.md, decision A) — per-nibble (E,F,I,T) soft labels + corpus ranking.

**Role (decision A)**: NOT a precise regression target — instead ① L_ground warm-start (annealed away) ② input for L_rank ranking supervision.
prosody DSP (AVD grounding) + SER (A/V/D, pluggable) + speech-act (directive/S/urgency/threat, pluggable) + cross-modal mismatch →
E/F/I/T. No speaker diarization (T3): entrainment → prosodic coherence, register → cross-modal (text↔prosody) mismatch.

Synthesis formulas (I gating is corrected to **nibble-level ¬coercion** — call-level ¬harm would zero out FSS-I, which is wrong):
  E = A · F = D·S·coldV·scam·rising_A_gate · I = D·warmV_txt·xmodal·subversion·(1−F_local) · T = coherence·balancedD·benign
Pure Python (numpy). If SER/speech-act are not injected, falls back to prosody/lexical proxies (can produce warm-start output before the gates).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

# Speech-act lexical proxies (when no speech-act is injected) — simple lexicon; the real teacher is a lightweight classifier (T2).
_DIRECTIVE = ("하세요", "하십시오", "해라", "해야", "요망", "바랍니다", "누르", "입력", "송금", "이체", "말씀")
_URGENCY = ("지금", "즉시", "바로", "빨리", "당장", "긴급", "오늘", "마감")
_THREAT = ("체포", "구속", "수사", "압류", "범죄", "처벌", "벌금", "고발", "검찰", "경찰")
_SCAM = ("계좌", "비밀번호", "인증", "명의", "대출", "수수료", "보증금", "안전계좌", "가상계좌", "앱", "설치")
_WARM = ("고객님", "도와", "안내", "친절", "걱정", "확인해", "감사", "네", "죄송")


def _clip01(x):
    return float(max(0.0, min(1.0, x)))


def _lex(text: str, words) -> float:
    if not text:
        return 0.0
    return _clip01(sum(text.count(w) for w in words) / 3.0)


def _avd_from_prosody(p: dict) -> tuple:
    """ProsodyFeatures dict → (A,V,D) proxies in [0,1]. AVD markers per docs/BASELINES.md."""
    A = _clip01(0.4 * _clip01(p.get("energy_mean", 0) * 8) + 0.3 * _clip01(p.get("f0_mean", 0) / 250.0)
                + 0.3 * _clip01(p.get("rate_proxy", 0) / 6.0))
    tilt = p.get("spectral_tilt", 0.0)                       # negative = low-frequency dominant = warm
    hnr = p.get("hnr_mean", 0.0)
    V = _clip01(0.5 + 0.5 * np.tanh(-tilt * 0.5) + 0.15 * np.tanh(hnr * 0.1))
    D = _clip01(0.4 * _clip01(p.get("f0_range", 0) / 120.0) + 0.3 * _clip01(p.get("energy_mean", 0) * 8)
                + 0.3 * (1.0 - _clip01(p.get("pause_ratio", 0))))
    return A, V, D


def nibble_signals(prosody: Optional[dict], text: str = "",
                   ser: Optional[tuple] = None,
                   speech_act: Optional[dict] = None) -> dict:
    """Per-nibble raw signals: A,V,D, directive/urgency/threat/subversion, xmodal_mismatch, F_local."""
    p = prosody or {}
    aP, vP, dP = _avd_from_prosody(p)
    A, V, D = ser if ser else (aP, vP, dP)                   # prefer SER when available (A/V); prosody dominates for D
    if ser:
        D = 0.5 * D + 0.5 * dP                               # no D labels → combine with prosody D
    sa = speech_act or {}
    directive = sa.get("directive", _lex(text, _DIRECTIVE))
    urgency = sa.get("urgency", _lex(text, _URGENCY))
    threat = sa.get("threat", _lex(text, _THREAT))
    subversion = sa.get("subversion", _lex(text, _SCAM))
    scam = _lex(text, _SCAM) if not sa else max(subversion, _lex(text, _SCAM))
    S = _clip01(max(directive, urgency, threat))
    coldV = 1.0 - V
    warm_txt = _lex(text, _WARM)                            # text warmth (lexical)
    xmodal = _clip01(max(0.0, warm_txt - V) * D)           # warm lexicon + dominant/cold tone = mismatch (I fingerprint)
    F_local = _clip01(D * S * coldV * max(scam, 0.2))
    return dict(A=A, V=V, D=D, directive=directive, urgency=urgency, threat=threat,
                subversion=subversion, scam=scam, S=S, coldV=coldV, warm_txt=warm_txt,
                xmodal=xmodal, F_local=F_local)


def _rising_gate(arousals: Sequence[float], i: int) -> float:
    """rising_A_gate: approaches 1 when arousal is rising up to time i (later > earlier). Captures pressure trajectories."""
    if i < 2:
        return 0.3
    early = np.mean(arousals[:max(1, i // 2)])
    now = np.mean(arousals[max(1, i // 2):i + 1])
    return _clip01(0.5 + (now - early) * 2.0)


def _coherence(arousals: Sequence[float], energies: Sequence[float]) -> float:
    """Prosodic coherence (speaker-agnostic T): low, stable arousal/energy variability → high. Coercion = divergence → low."""
    if len(arousals) < 2:
        return 0.5
    vol = float(np.std(arousals) + np.std(energies))
    return _clip01(1.0 - vol)


def emit(nibbles: List[dict], benign_prior: float = 0.5) -> List[Dict[str, float]]:
    """List of per-nibble raw signals → (T,I,F,E) soft label sequence. Reflects sequence context (rising, coherence).

    nibbles[i] = output of nibble_signals(...). benign_prior: corpus benign likelihood (ordinary conversation 1, FSS 0).
    """
    arous = [n["A"] for n in nibbles]
    energ = [n.get("A", 0) for n in nibbles]                 # energy proxy = A (simplification)
    coh = _coherence(arous, energ)
    out = []
    for i, n in enumerate(nibbles):
        rg = _rising_gate(arous, i)
        E = n["A"]
        F = _clip01(n["F_local"] * rg)
        I = _clip01(n["D"] * n["warm_txt"] * max(n["xmodal"], 0.1) * max(n["subversion"], 0.15) * (1.0 - n["F_local"]))
        balD = 1.0 - abs(n["D"] - 0.5) * 2.0                # balanced D is best
        T = _clip01(coh * _clip01(balD) * benign_prior)
        out.append({"T": T, "I": I, "F": F, "E": E})
    return out


# Corpus → benign_prior + rank tags (L_rank input). Ranking per docs/ARCHITECTURE.md: F fss>cc>benign etc.
CORPUS_PRIOR = {"fss": 0.0, "callcenter": 0.6, "benign": 1.0}


def _selftest() -> int:
    # FSS trajectory: early rapport (friendly + warm tone) → late pressure (directive + threat + rising arousal)
    early = nibble_signals({"energy_mean": 0.03, "f0_mean": 180, "f0_range": 40, "spectral_tilt": -1.0,
                            "pause_ratio": 0.2, "rate_proxy": 3},
                           text="고객님 안내 도와 드릴게요 확인해 주세요", speech_act={"subversion": 0.4})
    late = nibble_signals({"energy_mean": 0.12, "f0_mean": 240, "f0_range": 100, "spectral_tilt": 1.0,
                           "pause_ratio": 0.05, "rate_proxy": 6},
                          text="지금 즉시 검찰 수사 계좌 이체 하세요", speech_act={"subversion": 0.9})
    seq = emit([early, early, late, late], benign_prior=0.0)
    # early I > late I, late F > early F
    self_ok = seq[0]["I"] >= seq[-1]["I"] and seq[-1]["F"] > seq[0]["F"]
    assert self_ok, seq
    # High-arousal benign: arousal high but F must stay low (no arousal shortcut)
    exc = nibble_signals({"energy_mean": 0.15, "f0_mean": 250, "f0_range": 90, "spectral_tilt": -0.5,
                          "pause_ratio": 0.1, "rate_proxy": 6}, text="와 진짜 너무 좋다 대박 신난다")
    seqb = emit([exc, exc], benign_prior=1.0)
    assert seqb[0]["E"] > 0.4 and seqb[0]["F"] < 0.3, seqb
    print(f"[selftest] FSS 궤적 I(초{seq[0]['I']:.2f}>후{seq[-1]['I']:.2f})·F(후{seq[-1]['F']:.2f}>초{seq[0]['F']:.2f})")
    print(f"[selftest] 고각성benign E={seqb[0]['E']:.2f} F={seqb[0]['F']:.2f}(낮음=각성지름길 아님)")
    print("[selftest] 채널 teacher 합성 동작(I초반·F후반 궤적 + E⊥harm). warm-start/순위 입력용.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
