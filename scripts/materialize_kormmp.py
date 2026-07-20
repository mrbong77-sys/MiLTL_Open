#!/usr/bin/env python3
"""Materialize the KorMMP real-FSS pool (docs/BENCHMARK.md) — join real FSS harm + real benign
from case_inventory with ASR sidecar transcripts into a **compact inline bundle**.
Consumed via compose --inventory (real phishing prosody preserved).

Uses only information in the manifest: case_inventory_hard = {audio_path, class/label, source, meta}.
Transcripts are resolved via the ASR sidecar adjacent to the audio (asr_sidecar_path);
priority inline > text_path > sidecar, trying multiple tracks.
FSS sidecars live on the DGX -> running on the DGX materializes up to 506 harm cases.
Benign uses KsponSpeech sidecars.

  # DGX:
  python scripts/materialize_kormmp.py --inventory artifacts/manifest/case_inventory_hard.jsonl \
     --harm 506 --benign 600 --out artifacts/manifest/kormmp_real_full.jsonl
  # then:
  python scripts/canonical_bench.py --kormmp-inventory artifacts/manifest/kormmp_real_full.jsonl ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from miltl.baseline.asr_sidecar import asr_sidecar_path

_SUFFIXES = [".asr_{track}.txt", ".asr_light.txt", ".asr_000.txt"]   # try multiple suffixes to cover track-naming conventions


def _isharm(c):
    return c.get("class") == "harm" or int(c.get("label", 0)) == 1


def _audio(c):
    return c.get("audio_path") or c.get("audio_uri")


def _resolve_tx(c, track, mirror):
    """Resolve transcript: inline > text_path > ASR sidecar (multiple suffixes). '' if none."""
    if (c.get("transcript") or "").strip():
        return c["transcript"].strip()
    au = _audio(c)
    if au:
        for suf in [s.format(track=track) for s in _SUFFIXES]:
            try:
                t = asr_sidecar_path(au, suf, mirror).read_text(encoding="utf-8").strip()
                if t:
                    return t
            except OSError:
                pass
    tp = c.get("text_path")
    if tp:
        try:
            return Path(tp).read_text(encoding="utf-8").strip()
        except OSError:
            pass
    return ""


def run(inventory, track, mirror, n_harm, n_benign, lo, hi, out):
    harm, benign, seen = [], [], set()
    scanned = no_tx = 0
    with open(inventory, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                c = json.loads(ln)
            except json.JSONDecodeError:
                continue
            scanned += 1
            cid = c.get("case_id", c.get("call_id"))
            au = _audio(c)
            if not au or cid in seen:
                continue
            h = _isharm(c)
            if h and "fss" not in str(au).lower():          # harm must be real FSS only
                continue
            if (h and len(harm) >= n_harm) or ((not h) and len(benign) >= n_benign):
                if len(harm) >= n_harm and len(benign) >= n_benign:
                    break
                continue
            tx = _resolve_tx(c, track, mirror)
            nw = len(tx.split())
            if not tx or nw < lo or nw > hi:
                no_tx += 1
                continue
            seen.add(cid)
            rec = {"case_id": cid, "class": "harm" if h else "benign", "label": 1 if h else 0,
                   "source": c.get("source", "?"), "transcript": tx, "audio_uri": au,
                   "meta": {"n_words": nw, **(c.get("meta") or {})}}
            (harm if h else benign).append(rec)
            if len(harm) >= n_harm and len(benign) >= n_benign:
                break
    rows = harm + benign
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"[materialize] harm={len(harm)} benign={len(benign)} → {out} "
          f"(scanned {scanned}, transcript-missing/len-filtered {no_tx})", flush=True)
    if not harm:
        print("  ⚠️ harm=0 — FSS ASR sidecars are not present in this environment (DGX only). Run on the DGX.", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default="artifacts/manifest/case_inventory_hard.jsonl")
    ap.add_argument("--asr-track", default="000")
    ap.add_argument("--mirror", default="artifacts/asr")
    ap.add_argument("--harm", type=int, default=506)
    ap.add_argument("--benign", type=int, default=600)
    ap.add_argument("--len-lo", type=int, default=150)
    ap.add_argument("--len-hi", type=int, default=360)
    ap.add_argument("--out", default="artifacts/manifest/kormmp_real_full.jsonl")
    args = ap.parse_args()
    return run(args.inventory, args.asr_track, args.mirror, args.harm, args.benign,
               args.len_lo, args.len_hi, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
