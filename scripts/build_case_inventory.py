#!/usr/bin/env python3
"""Case inventory builder (DGX) — enumerate each source in datasets.yaml via its adapter → case-level meta jsonl (docs/BENCHMARK.md).

Input to compose_testset. **Metadata only** (paths, modality, class, tags, word counts) — transcript
content is never stored (PII, git-safe).
ADAPTERS[adapter](root) yields Call (call_id, label, utterances, audio_path, split_keys) → case_row conversion.
call_center (the build_benchmark path) sits outside the adapters, so it is handled separately
(--include-call-center is future work).

  python scripts/build_case_inventory.py --out artifacts/manifest/case_inventory.jsonl
  python scripts/build_case_inventory.py --sources fss emotion_dialog --limit 500
⚠️ Requires real data (DGX). The core conversion (case_row) is verified DGX-free by tests/test_case_inventory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

_META_KEYS = ("emotion", "anger_ratio", "style", "domain", "topic")   # tagging-type keys among split_keys


def _audio_meta(audio, sr=16000):
    """Audio file → (duration_s). DGX (decode). 0 on failure."""
    if not audio:
        return 0.0
    try:
        from miltl.nibble.audio_decode import decode_to_pcm
        x, s = decode_to_pcm(str(audio), sr=sr)
        return round(len(x) / s, 2) if s else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def case_row(call, klass: str, with_audio_meta: bool = False) -> dict:
    """Call → one case-inventory row (metadata only). modality = audio presence; tags = tagging-type split_keys.
    with_audio_meta=True extracts duration/SNR/speaker count (DGX; audio decode + label meta)."""
    has_audio = bool(getattr(call, "audio_path", None) or getattr(call, "audio_paths", None))
    utts = list(getattr(call, "utterances", []) or [])
    has_text = any(bool(u and u.strip()) for u in utts)
    modality = ("dual" if has_audio and has_text else "wave" if has_audio else "text")
    sk = getattr(call, "split_keys", {}) or {}
    tags = {k: sk[k] for k in _META_KEYS if k in sk}
    n_words = sum(len(u.split()) for u in utts) if has_text else 0
    audio = getattr(call, "audio_path", None)
    if not audio:
        aps = getattr(call, "audio_paths", None)
        audio = aps[0] if aps else None
    row = {
        "case_id": call.call_id, "source": getattr(call, "source", "?"), "class": klass,
        "modality": modality, "tags": tags, "n_words": n_words,
        "audio_path": audio, "n_audio": len(getattr(call, "audio_paths", None) or ([audio] if audio else [])),
        # SNR / speaker count: taken from split_keys/meta when present (emotion-data json). Defaults otherwise.
        "snr_db": sk.get("snr_db"), "n_speakers": int(sk.get("n_speakers", 0) or 0),
        "duration_s": 0.0,
    }
    if with_audio_meta and audio:
        row["duration_s"] = _audio_meta(audio)
    return row


def _flatten(reg):
    for group, members in reg.items():
        if isinstance(members, dict):
            for key, spec in members.items():
                if isinstance(spec, dict) and "adapter" in spec:
                    yield group, key, spec


def main() -> int:
    ap = argparse.ArgumentParser(description="Case inventory builder (DGX)")
    ap.add_argument("--sources", nargs="*", help="restrict to specific adapter keys (default: all)")
    ap.add_argument("--limit", type=int, help="max cases per source")
    ap.add_argument("--out", default="artifacts/manifest/case_inventory.jsonl")
    ap.add_argument("--audio-meta", action="store_true",
                    help="extract audio duration/SNR/speaker count (DGX, slow — decodes). For length control and SNR probes.")
    args = ap.parse_args()

    from miltl.nibble import ADAPTERS
    reg = yaml.safe_load(Path("configs/datasets.yaml").read_text(encoding="utf-8"))

    from collections import Counter
    stats = Counter()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for group, key, spec in _flatten(reg):
            adapter = spec["adapter"]
            if adapter not in ADAPTERS:
                continue                                   # outside the adapters (e.g. call_center) → skip
            if args.sources and adapter not in args.sources and key not in args.sources:
                continue
            klass = "harm" if spec.get("label") == "phishing" else "benign"
            root = spec.get("root")
            try:
                print(f"[inv] ▶ {group}/{key} ({adapter}) scan started …", flush=True)
                it = ADAPTERS[adapter](root) if root else ADAPTERS[adapter]()
                cnt = 0
                for call in it:
                    row = case_row(call, klass, with_audio_meta=args.audio_meta)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
                    stats[f"{klass}/{row['modality']}"] += 1
                    n += 1
                    cnt += 1
                    if cnt % 500 == 0:                    # per-source progress (keep the log moving)
                        print(f"[inv]   {group}/{key} … {cnt} cases (total {n})", flush=True)
                    if args.limit and cnt >= args.limit:
                        break
                print(f"[inv] ✓ {group}/{key} ({adapter}) → {cnt} cases (total {n})", flush=True)
            except Exception as e:  # noqa: BLE001 — skip sources without data (with logging)
                print(f"[inv] ⚠️ {group}/{key} ({adapter}) skipped: {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
    print(f"[inv] total {n} cases → {args.out}")
    print(f"[inv] distribution: {dict(stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
