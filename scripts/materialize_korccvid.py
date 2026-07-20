#!/usr/bin/env python3
"""Materialize a KorCCViD test bundle for the uniform harness (docs/BENCHMARK.md).

KorCCViD is a **transcript-only corpus** (no audio, fetch_korccvi.py). MiLTL's cross-modal XM and
prosody channels therefore fall back to text (inert), and KorCCViD is only meaningful as a
contrast bench showing "standard-corpus classification = trivial" (all models high, undifferentiated).
This script converts test_pool.jsonl into the bench_hard_slices/gate2_cascade consumption schema
({call_id, transcript, audio_uri:null, label, slice, meta{n_words}}) -> **same result-sheet
fidelity as KorMMP** (except channels have no audio = XM≈0 text fallback).
Not a hard slice -> slice=korccvid-*, decorrelated=n/a (standard).

  python scripts/materialize_korccvid.py \
     --in artifacts/frozen/korccvid/test_pool.jsonl --out artifacts/manifest/korccvid_test_bundle.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="artifacts/frozen/korccvid/test_pool.jsonl")
    ap.add_argument("--out", default="artifacts/manifest/korccvid_test_bundle.jsonl")
    ap.add_argument("--min-words", type=int, default=0, help="drop calls below this word count (0=keep all)")
    ap.add_argument("--total", type=int, default=0,
                    help="cap to N calls, label-balanced (0=all 879). Use for slow detectors like B3-Bllossom.")
    ap.add_argument("--harm-ratio", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    for r in rows:
        tr = (r.get("transcript") or "").strip()
        nw = len(tr.split())
        if nw < args.min_words:
            continue
        lab = int(r.get("label", 0))
        out.append({"call_id": r.get("call_id"), "label": lab, "source": r.get("source", "korccvi"),
                    "split": "test", "transcript": tr,
                    "audio_uri": None,                       # KorCCViD = transcript-only (no audio)
                    "slice": "korccvid-harm" if lab else "korccvid-benign",
                    "meta": {"n_words": nw, "density": 0.0}})
    if args.total and len(out) > args.total:                 # label-balanced random subsample (budget for slow detectors like B3)
        import random
        rng = random.Random(args.seed)
        harm = [r for r in out if r["label"] == 1]; ben = [r for r in out if r["label"] == 0]
        nh = min(len(harm), int(round(args.total * args.harm_ratio))); nb = min(len(ben), args.total - nh)
        rng.shuffle(harm); rng.shuffle(ben)
        out = harm[:nh] + ben[:nb]; rng.shuffle(out)
        print(f"[korccvid] capped to {len(out)} calls (harm={nh} benign={nb}, seed={args.seed})", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n",
                              encoding="utf-8")
    pos = sum(r["label"] for r in out)
    print(f"[korccvid] {len(out)} calls -> {args.out} (harm={pos} benign={len(out)-pos}). "
          f"NOTE: transcript-only corpus (no audio) -> MiLTL XM/prosody = text-fallback (inert).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
