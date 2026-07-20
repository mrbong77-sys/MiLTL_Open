#!/usr/bin/env python3
"""Lexical-shortcut measurement (docs/BENCHMARK.md) — show that legacy text models reduce to
scam-lexicon correlation.

KorMMP AUROC of a score built only from the scam/threat/directive/urgency lexicons =
**upper bound of the legacy shortcut**.
Plus, the lexical-density distribution quantifies hard-slice (lexical decorrelation) availability:
  hard-harm = low-lexical-density phishing (legacy FN) · hard-benign = high-lexical-density benign (legacy FP).

DGX-free: uses *_full.jsonl (inline transcript).
  python scripts/lexical_shortcut.py --bundle artifacts/manifest/kormmp_full.jsonl
  python scripts/lexical_shortcut.py --bundle artifacts/manifest/kormmp_hard_full.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from miltl.native.channel_teacher import _SCAM, _THREAT, _DIRECTIVE, _URGENCY
from scripts.train_channel_extractors import _auroc


def _hits(text, words):
    return sum(text.count(w) for w in words)


def _lex_scores(text):
    """Lexical score components + composite (the signal legacy models consume)."""
    scam = _hits(text, _SCAM); threat = _hits(text, _THREAT)
    directive = _hits(text, _DIRECTIVE); urgency = _hits(text, _URGENCY)
    total = scam + threat + directive + urgency
    return {"scam": scam, "threat": threat, "directive": directive, "urgency": urgency,
            "harm_lex": scam * 1.5 + threat * 1.5 + directive + urgency}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in Path(args.bundle).read_text(encoding="utf-8").splitlines() if l.strip()]
    y, scores, dens, srcs = [], [], [], []
    for r in rows:
        t = r.get("transcript") or ""
        lab = int(r.get("label", r.get("harm", 0)))
        ls = _lex_scores(t)
        nw = max(len(t.split()), 1)
        y.append(lab); scores.append(ls["harm_lex"]); dens.append(ls["harm_lex"] / nw * 100)
        srcs.append(r.get("source", "?"))
    y = np.array(y); scores = np.array(scores, float); dens = np.array(dens, float)

    au = _auroc(scores, y)
    print(f"\n=== Lexical shortcut (bundle={Path(args.bundle).name}, n={len(y)}, harm={int(y.sum())}) ===", flush=True)
    print(f"  scam-lexicon composite score AUROC→harm = {au:.3f}   ← legacy text-shortcut upper bound", flush=True)
    print(f"  harm  lexical density (hits/100 words) med={np.median(dens[y==1]):.2f}  mean={dens[y==1].mean():.2f}", flush=True)
    print(f"  benign lexical density                 med={np.median(dens[y==0]):.2f}  mean={dens[y==0].mean():.2f}", flush=True)

    # Hard-slice availability: quantify the decorrelated region via lexical density
    ben_hi = np.percentile(dens[y == 0], 90)         # upper threshold of benign lexical density
    harm_lo = np.percentile(dens[y == 1], 25)        # lower threshold of phishing lexical density
    hard_harm = int(((y == 1) & (dens <= ben_hi)).sum())     # phishing with benign-level lexicon = legacy FN candidates
    hard_benign = int(((y == 0) & (dens >= harm_lo)).sum())  # benign with phishing-level lexicon = legacy FP candidates
    print(f"\n  [Hard-slice availability — lexical decorrelation]", flush=True)
    print(f"  hard-harm (phishing & density<=benign 90%tile={ben_hi:.2f}) = {hard_harm}/{int(y.sum())}  (legacy FN candidates)", flush=True)
    print(f"  hard-benign (benign & density>=phishing 25%tile={harm_lo:.2f}) = {hard_benign}/{int((y==0).sum())}  (legacy FP candidates)", flush=True)

    # Per-source lexical density (where do the trap benigns live?)
    from collections import defaultdict
    bysrc = defaultdict(list)
    for s, d, lab in zip(srcs, dens, y):
        bysrc[(s, lab)].append(d)
    print(f"\n  [Source x label lexical density]", flush=True)
    for k in sorted(bysrc):
        v = bysrc[k]
        print(f"    {k[0]:<18} label={k[1]}  n={len(v):<3} lexical-density med={np.median(v):.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
