"""XAI explanation generator — channel signals + final verdict → user-facing reasons and recommended actions (docs/BENCHMARK.md).

Qwen(0.5B) handles yes/no verdicts well with SFT but is unstable at generating rationales → a **deterministic sanitize algorithm**
assembles reasons from channel-bottleneck signals (XM, F, I, E, cold/warmth) + lexical cues in the transcript. Always consistent, interpretable, reproducible.
On edge deployment: report reasons to the user → grounds for actions such as ending the call or alerting a guardian.
"""
from __future__ import annotations

from typing import Optional

# Thresholds (conditions for uttering a reason on edge) — conservative (suppress over-alerting). Values are based on channel distributions (docs/ARCHITECTURE.md).
_XM_HI, _F_HI, _I_HI, _E_HI = 0.10, 0.25, 0.62, 0.65

_ACTIONS = {
    "harm": ["Phishing suspected — display warning to user",
             "Recommend ending the call if sensitive info / bank transfer is requested",
             "Advise reporting to guardian / authorities (112, 1332)"],
    "caution": ["Caution — notify user of suspicious signals",
                "Recommend immediate stop if personal info / authentication / transfer is requested"],
    "benign": ["Judged a normal call — no action needed"],
}


def _scan_cues(transcript: str) -> list:
    """Extract phishing-typical lexical cues from the transcript (impersonation, threat, directives, urgency). Evidence for user explanation. Quoted words stay in original Korean."""
    if not transcript:
        return []
    from miltl.native.channel_teacher import _SCAM, _THREAT, _DIRECTIVE, _URGENCY
    cues = []
    for label, lex in (("impersonation/financial & personal-info", _SCAM),
                       ("threat/investigation", _THREAT),
                       ("action directives", _DIRECTIVE), ("urgency pressure", _URGENCY)):
        hits = [w for w in lex if w in transcript]
        if hits:
            cues.append(f"{label} ('{'·'.join(hits[:3])}')")
    return cues


def explain_decision(diag: dict, decision: str, transcript: str = "",
                     p1: Optional[float] = None) -> dict:
    """Channel diagnostics + final verdict → {verdict, reasons[], action[], summary}. Deterministic, reproducible.

    decision: 'harm'|'benign' (cascade final). reasons contains only fired signals (suppress over-explanation).
    """
    xm = float(diag.get("XM", 0.0)); F = float(diag.get("F", 0.0)); I = float(diag.get("I", 0.0))
    E = float(diag.get("E", 0.0)); cold = float(diag.get("cold", 0.0)); warmth = float(diag.get("warmth", 0.0))
    is_harm = str(decision).startswith("harm")
    reasons = []
    if xm >= _XM_HI:
        reasons.append(f"wording is warm/reassuring (warmth {warmth:.2f}) but the voice is cold/controlling "
                       f"(cold {cold:.2f}) — large speech-voice mismatch (XM {xm:.2f}), a deception "
                       f"(social-engineering) signature")
    if F >= _F_HI:
        reasons.append(f"coercive/pressuring phrasing (F {F:.2f})")
    if I >= _I_HI:
        reasons.append(f"latent threat accumulates as the call proceeds (I {I:.2f})")
    if E >= _E_HI:
        reasons.append(f"tense/aroused vocal tone (E {E:.2f})")
    cues = _scan_cues(transcript)
    if cues:
        reasons.append("transcript contains phishing-typical phrases: " + " · ".join(cues))

    if is_harm:
        verdict = "HIGH-RISK (phishing suspected)"
        if not reasons:
            reasons.append("combined risk signals exceed the normal range")
        action = _ACTIONS["harm"]
    else:
        verdict = "SAFE (normal call)"
        # If benign but some signals fired, promote to 'caution' (edge-safe side)
        if reasons:
            verdict = "SAFE (minor caution)"
            action = _ACTIONS["caution"]
        else:
            reasons.append("speech and voice tone are congruent; no threat, transfer-inducement, or "
                           "impersonation signs")
            action = _ACTIONS["benign"]

    conf = f"(Gate-1 risk {p1:.2f}) " if p1 is not None else ""
    body = "; ".join(reasons)
    summary = (f"{'[HIGH-RISK]' if is_harm else '[SAFE]'} This call is classified as **{verdict}** "
               f"{conf}because: {body}. -> {action[0]}")
    return {"verdict": verdict, "reasons": reasons, "action": action, "summary": summary}
