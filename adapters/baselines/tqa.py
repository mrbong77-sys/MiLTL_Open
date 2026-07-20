"""TQA — Threat Question Answering (see docs/ARCHITECTURE.md) — a harm-domain substitution of RLDX-1 robot-VQA.

Just as VQA adapts a VLM toward extracting *action-relevant* features, TQA adapts and grounds Gate-2 (Qwen)
toward *harm-relevant* ones. Nibble features **route only the questions that fit the situation** -> even a
small Qwen makes deterministic, fast decisions. Pure (DGX-free); gate2_slm injects them into the prompt for
both training and inference.

Usage:
  feat = nibble_features(text_nibbles, wave_nibbles, p1, state)   # gate2_slm
  qs = select_tqa(feat)                                            # situation-specific questions
  # inject via gate2_slm.prompt(..., tqa=qs)
"""
from __future__ import annotations

from typing import Dict, List, Optional

# Question bank per threat category (steers attention onto harm signals). Core is always included.
TQA_BANK: Dict[str, List[str]] = {
    "core": [
        "이 통화에 자금 이체·송금·현금 전달 요구가 있는가?",
        "수사기관·금융기관·공공기관 사칭 정황이 있는가?",
    ],
    "app_remote": ["앱 설치·원격제어·링크 클릭 유도가 있는가?"],
    "pii": ["개인정보·주민등록번호·인증번호·비밀번호를 요구하는가?"],
    "pressure": ["공포 조성·긴급성 압박·비밀 유지 강요가 있는가?"],
    "ramp": ["대화 후반부로 갈수록 위협 강도(위협에너지 E·유해 F)가 급증하는가?"],
    "hardneg": ["정상 금융·업무 상담으로 보이지만, 실제로는 이체·개인정보 탈취 유도가 숨어 있는가?"],
    "prosody": ["전사가 불완전한 상태에서, 말의 압박·긴장 운율만으로 위협이 감지되는가?"],
    "final": ["위 정황을 종합할 때, 이 통화는 실제 위해로 이어지는 보이스피싱(harm)인가? (예/아니오)"],
}


def select_tqa(features: dict, max_q: int = 6,
               kw_present: Optional[bool] = None) -> List[str]:
    """Nibble features -> situation-appropriate question selection (always core+final, conditional categories added)."""
    text, wave = features.get("text"), features.get("wave")
    state = features.get("state")
    cats = ["core"]
    # High harm/threat energy -> pressure, PII, and app questions
    F = (text or {}).get("F", 0.0); E = (text or {}).get("E", 0.0); fe = (text or {}).get("fe", 0.0)
    if F >= 0.2 or E >= 0.15:
        cats += ["pressure", "pii", "app_remote"]
    if fe >= 0.1 or (text or {}).get("ramp", 0.0) > 0.05:      # Pressure+energy co-occurrence / late-half surge
        cats += ["ramp"]
    if kw_present:                                             # Financial vocabulary present -> hard-negative discrimination
        cats += ["hardneg"]
    if state == "WAVE-ONLY" or text is None:                  # Text missing / ASR lag -> prosody
        cats += ["prosody"]
    # De-duplicate while preserving order: core, then conditional, final last
    seen, qs = set(), []
    for c in cats:
        for q in TQA_BANK.get(c, []):
            if q not in seen:
                seen.add(q); qs.append(q)
    qs = qs[:max(1, max_q - len(TQA_BANK["final"]))] + TQA_BANK["final"]
    return qs
