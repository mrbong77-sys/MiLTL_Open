#!/usr/bin/env python3
"""Rebuild demo/cases_canonical.json from the canonical benchmark artifacts.

Selects 20 archetype cases (legacy home turf / easy-harm / FSS hard-harm /
8 hard-benign / 7 hard-harm) whose transcripts and per-detector verdicts are
quoted VERBATIM from artifacts/rounds/canonical/ (bundles + sheets). Every
verdict shown in the demo is a recorded benchmark outcome, not a live re-run.

License gate: only synth (self-authored), FSS (KOGL Type-1) and KorCCViD
(CC BY-NC-SA 4.0) transcripts are eligible — AI-Hub-derived rows (emotion_dialog)
are never selected (their transcripts are redacted in the public artifacts).

  python demo/build_cases.py            # rewrites demo/cases_canonical.json
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "artifacts" / "rounds" / "canonical"
OUT = Path(__file__).resolve().parent / "cases_canonical.json"

# sheet detector name -> display name (matches the demo verdict panel)
DET = {
    "MiLTL-Cascade": "MiLTL-Cascade",
    "hf(frozen)": "hf-encoder",
    "tree(frozen)": "tree",
    "cnn_bilstm(frozen)": "cnn-bilstm",
    "lexical(text-proxy)": "lexical-proxy",
    "Bllossom-B3(public-finetuned)": "Bllossom-B3",
    "Wave-Seq(audio-only)": "Wave-Seq(audio-only)",
    "MiLTL-Dual(naive-fusion text+wave)": "MiLTL-Dual(naive fusion)",
}

HB_TITLES = {"은행": "Bank consultation (legitimate)", "카드": "Credit card consultation (legitimate)",
             "보험": "Insurance consultation (legitimate)", "통신": "Telecom consultation (legitimate)",
             "배송": "Delivery/courier consultation (legitimate)", "관공서": "Government office consultation (legitimate)",
             "병원": "Hospital consultation (legitimate)", "증권": "Securities consultation (legitimate)"}
# Up to 7 hard-harm scenarios are taken from this pool, in order, skipping any scenario that
# has no legacy-failure archetype in the canonical run (all detectors correct there).
HH_TITLES = {"가족사칭": "Family impersonation — no scam keywords", "기관사칭": "Institution impersonation — no scam keywords",
             "대출사칭": "Loan scam — no scam keywords", "환급사칭": "Refund scam — no scam keywords",
             "투자리딩": "Investment-coaching scam — no scam keywords", "지원금사칭": "Subsidy scam — no scam keywords",
             "택배사칭": "Delivery scam — no scam keywords", "지인사칭": "Acquaintance impersonation — no scam keywords"}
HH_MAX = 7


def _load(corpus):
    """(bundle rows by cid, per-call detector rows) pooled over all seeds."""
    calls, sheets = {}, defaultdict(dict)
    for bf in sorted(CANON.glob(f"bundle_{corpus}_*.jsonl")):
        seed = int(bf.stem.split("_")[-1])
        for line in open(bf, encoding="utf-8"):
            r = json.loads(line)
            r["_seed"] = seed
            calls[(seed, r["call_id"])] = r
    for sf in sorted(CANON.glob(f"sheet_{corpus}_*.csv")):
        seed = int(sf.stem.split("_")[-1])
        for row in csv.DictReader(open(sf, encoding="utf-8-sig")):
            sheets[(seed, row["call_id"])][row["detector"]] = row
    return calls, sheets


def _case(call, dets, title, group, prosody, why, corpus):
    cas = dets["MiLTL-Cascade"]
    verd = {}
    for raw, name in DET.items():
        if raw in dets:
            d = dets[raw]
            try:
                score = round(float(d["score"]), 3)
            except ValueError:
                score = d["score"]
            verd[name] = {"outcome": d["outcome"], "score": score}
    return {
        "cid": call["call_id"].replace("korccvi_", "korccvi:"), "seed": call["_seed"],
        "label": int(call["label"]), "slice": call.get("slice", ""),
        "density": round(float(call.get("meta", {}).get("density", 0.0)), 2),
        "channels": {k: round(float(cas[k]), 3) for k in ("T", "I", "F", "E", "XM")},
        "p1": round(float(cas["p1"]), 3), "band": cas["band"],
        "transcript": call["transcript"], "verdicts": verd,
        "title": title, "group": group, "prosody": prosody, "why": why, "corpus": corpus,
    }


def _ok(dets, det, outcome):
    return det in dets and dets[det]["outcome"] == outcome


def main():
    cases = []
    km_calls, km_sheets = _load("kormmp")
    kc_calls, kc_sheets = _load("korccvid")

    def candidates(calls, sheets, pred):
        out = []
        for key, call in calls.items():
            dets = sheets.get(key, {})
            if "MiLTL-Cascade" not in dets or not call.get("transcript"):
                continue
            if pred(call, dets):
                out.append((key, call, dets))
        return sorted(out, key=lambda x: x[0][1])   # deterministic (call_id)

    # 1) KorCCViD home turf x2 — every legacy detector correct (why standard corpus misleads)
    home = candidates(kc_calls, kc_sheets, lambda c, d: int(c["label"]) == 1 and
                      all(_ok(d, k, "TP") for k in ("MiLTL-Cascade", "hf(frozen)", "tree(frozen)",
                                                    "cnn_bilstm(frozen)", "Bllossom-B3(public-finetuned)")))
    pros = [c for c in home if "검" in c[1]["transcript"][:200]]
    poli = [c for c in home if "경찰" in c[1]["transcript"][:200]]
    picks = [(pros or home)[0], (poli or [h for h in home if h != (pros or home)[0]])[0]]
    whys = ["Textbook KorCCViD scam. On their OWN training corpus every legacy detector is correct (TP) — "
            "hf/cnn/tree memorized it. That in-corpus perfection is why a KorCCViD-only score is misleading.",
            "Another standard-corpus scam — all legacy detectors correct. Compare with the hard slices below, "
            "where the same detectors flip."]
    titles = ["Prosecutor impersonation (standard corpus)", "Police impersonation (standard corpus)"]
    for (key, call, dets), t, w in zip(picks, titles, whys):
        cases.append(_case(call, dets, t, "Legacy home turf · corpus classifiers look flawless", "none", w, "KorCCViD"))

    # 2) Easy-harm (real FSS, keyword-dense) — even the lexical matcher fires
    easy = candidates(km_calls, km_sheets, lambda c, d: c.get("source") == "fss_audio" and
                      c.get("slice") == "easy-harm" and _ok(d, "lexical(text-proxy)", "TP") and
                      _ok(d, "MiLTL-Cascade", "TP"))
    key, call, dets = easy[0]
    cases.append(_case(call, dets, "Impersonation, scam-vocabulary-heavy (real FSS)",
                       "Easy-harm · keyword-dense — even the lexical proxy fires", "cold",
                       "Real FSS phishing packed with scam vocabulary. When the scam words are present even the "
                       "pure keyword matcher is right — the hard slices below remove that crutch.", "KorMMP"))

    # 3) FSS hard-harm x2 — (a) a big-model miss (B3 or Wave FN), (b) tree miss while hf correct
    fss_hard = candidates(km_calls, km_sheets, lambda c, d: c.get("source") == "fss_audio" and
                          "hard-harm" in c.get("slice", "") and _ok(d, "MiLTL-Cascade", "TP"))
    a = [c for c in fss_hard if _ok(c[2], "Bllossom-B3(public-finetuned)", "FN") or _ok(c[2], "Wave-Seq(audio-only)", "FN")]
    b = [c for c in fss_hard if _ok(c[2], "tree(frozen)", "FN") and _ok(c[2], "hf(frozen)", "TP") and c != (a or fss_hard)[0]]
    key, call, dets = (a or fss_hard)[0]
    cases.append(_case(call, dets, "Real vishing recording (raw ASR)", "Hard-harm · real vishing audio — a large model misses it",
                       "cold", "Genuine noisy phone audio, low scam-word density. A large detector misses it (FN); "
                       "MiLTL catches it from the channel signals.", "KorMMP"))
    key, call, dets = (b or [c for c in fss_hard if c != (a or fss_hard)[0]])[0]
    cases.append(_case(call, dets, "Low-pressure scam (real FSS)", "Hard-harm · exposes the weak tree ensemble", "cold",
                       "Real FSS phishing that the frozen CatBoost tree misses while the hf-encoder still catches it — "
                       "detector quality differs even inside the legacy family.", "KorMMP"))

    # 4) Synthetic hard-benign x8 — legitimate finance-adjacent calls, legacy false-positives
    for scen, title in HB_TITLES.items():
        hb = candidates(km_calls, km_sheets, lambda c, d, s=scen: c.get("source") == "synth" and
                        c.get("slice") == "synth-hard-benign" and
                        (c.get("scenario") or c.get("meta", {}).get("scenario")) == s and
                        _ok(d, "MiLTL-Cascade", "TN") and
                        sum(_ok(d, k, "FP") for k in ("hf(frozen)", "tree(frozen)", "cnn_bilstm(frozen)",
                                                      "Bllossom-B3(public-finetuned)")) >= 2)
        if not hb:
            continue
        hb.sort(key=lambda x: -sum(_ok(x[2], k, "FP") for k in ("hf(frozen)", "tree(frozen)",
                                                                "cnn_bilstm(frozen)", "Bllossom-B3(public-finetuned)")))
        key, call, dets = hb[0]
        nfp = sum(1 for v in dets.values() if v["outcome"] == "FP")
        cases.append(_case(call, dets, title, "Hard-benign · legitimate finance call — legacy false-positives", "warm",
                           f"Legitimate consultation full of finance/authority vocabulary — {nfp} legacy detectors "
                           "false-positive ('finance words = scam'). MiLTL reads the cooperative channel pattern and stays benign.",
                           "KorMMP"))

    # 5) Synthetic hard-harm (up to HH_MAX) — scam-word-free grooming that big/audio models miss
    hh_taken = 0
    for scen, title in HH_TITLES.items():
        if hh_taken >= HH_MAX:
            break
        hh = candidates(km_calls, km_sheets, lambda c, d, s=scen: c.get("source") == "synth" and
                        c.get("slice") == "synth-hard-harm" and
                        (c.get("scenario") or c.get("meta", {}).get("scenario")) == s and
                        _ok(d, "MiLTL-Cascade", "TP") and
                        (_ok(d, "Bllossom-B3(public-finetuned)", "FN") or _ok(d, "Wave-Seq(audio-only)", "FN")
                         or _ok(d, "tree(frozen)", "FN") or _ok(d, "hf(frozen)", "FN")))
        if not hh:
            continue
        hh.sort(key=lambda x: (-int(_ok(x[2], "Bllossom-B3(public-finetuned)", "FN")),
                               -sum(1 for v in x[2].values() if v["outcome"] == "FN")))
        key, call, dets = hh[0]
        miss = [DET[k] for k in DET if _ok(dets, k, "FN")]
        cases.append(_case(call, dets, title, "Hard-harm · scam-word-free grooming — big models miss it", "cold-pressure",
                           f"Vocabulary-free grooming (density {float(call.get('meta', {}).get('density', 0)):.1f}): "
                           f"{', '.join(miss)} miss it (FN). MiLTL catches it on arousal (E) + cross-modal contradiction (XM).",
                           "KorMMP"))
        hh_taken += 1

    seeds = json.load(open(CANON / "seeds.json"))["seeds"]
    doc = {
        "cases": cases,
        "provenance": ("Transcripts and per-detector verdicts are quoted verbatim from the 5-seed canonical "
                       f"benchmark (seeds {'/'.join(str(s) for s in seeds)}). Outcomes: TP/TN correct, FP/FN wrong."),
        "sources": ("Synthetic cases self-authored (Apache-2.0); FSS cases KOGL Type-1 (금융감독원); "
                    "KorCCViD cases CC BY-NC-SA 4.0. No AI-Hub-derived transcripts."),
    }
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT.name}: {len(cases)} cases "
          f"({sum(1 for c in cases if c['corpus'] == 'KorCCViD')} KorCCViD, "
          f"{sum(1 for c in cases if c['corpus'] == 'KorMMP')} KorMMP)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
