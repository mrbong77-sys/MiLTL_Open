#!/usr/bin/env python3
"""Gate-2 SLM LoRA SFT — training on the channel-bottleneck escalate band (docs/BENCHMARK.md) — DGX.

Fixes the problem that zero-shot Qwen0.5B fails to separate the escalate band (low-XM phishing vs
cold normal calls), degenerating into parroting (see docs/BENCHMARK.md).
Freeze protocol: training = **KorCCViD + synthetic (seed≠eval)** bundles (NOT KorMMP). Evaluation = KorMMP (gate2_cascade).

Flow: train bundle → Gate-1 (channel bottleneck) scoring and banding → only the **escalate band**
  becomes {transcript, diag, label} examples
  → Gate2SLM.fit_channels(LoRA SFT, transcript-first + raw-channel prompt) → save adapter. Evaluation is separate (cascade).

  # DGX (training) — KorCCViD trainset (+optional synthetic seed99):
  python scripts/gate2_sft.py --train-bundle artifacts/manifest/native_trainset.jsonl \
     --synth-bundle artifacts/rounds/synth_99.jsonl \
     --ckpt artifacts/models/channel_extractors.pt --head artifacts/models/miltl_head.pt \
     --gate2-model Qwen/Qwen2.5-0.5B-Instruct --out artifacts/models/gate2_adapter \
     --epochs 3 --device cuda
  # Evaluate after training (adapter loaded):
  python scripts/gate2_cascade.py --bundle artifacts/rounds/hardX_42.jsonl \
     --ckpt … --head … --gate2-adapter artifacts/models/gate2_adapter --device cuda --out-ledger …
  # DGX-free validation (SLM not loaded, escalate example-build stats only):
  python scripts/gate2_sft.py --train-bundle <bundle> --ckpt … --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def _row_to_call(r):
    """Normalize a bundle row — works for both eval bundles (transcript/audio_uri/label) and native_trainset (utterances/audio_uris/harm)."""
    tr = r.get("transcript") or " ".join(r.get("utterances", []) or [])
    au = r.get("audio_uri") or r.get("audio_path") or (r.get("audio_uris") or [None])[0]
    lab = r.get("label", r.get("harm"))
    if lab is None:
        lab = 1 if r.get("class") == "harm" else 0
    return SimpleNamespace(transcript=tr, audio_uri=au, label=int(lab))


def build_examples(bundles, ckpt, head, device, tau_low, tau_high, blend, anchor, band,
                   codec_equalize=False):
    """train bundle(s) → Gate-1 banding → {transcript, diag, label} examples for the target band + band stats.

    codec_equalize=True: extract diag with the same channel equalization as evaluation (bench --fair-audio)
    → train/eval distribution alignment.
    """
    rows = []
    for b in bundles:
        if b and Path(b).is_file():
            rows += [json.loads(l) for l in Path(b).read_text(encoding="utf-8").splitlines() if l.strip()]
    calls = [_row_to_call(r) for r in rows]
    from adapters.baselines.native_channel import ChannelBottleneckDetector
    g1 = ChannelBottleneckDetector(ckpt=ckpt, head=head, device=device, channels="calib",
                                   blend_analytic=blend, anchor_words=anchor, codec_equalize=codec_equalize)
    exs, stats = [], {"benign": 0, "escalate": 0, "harm": 0}
    for c in calls:
        p1 = float(g1.score(c))
        diag = dict(getattr(g1, "last_diag", {}))
        b = "benign" if p1 <= tau_low else "harm" if p1 >= tau_high else "escalate"
        stats[b] += 1
        if band == "escalate" and b != "escalate":
            continue
        if band == "confident" and b == "escalate":
            continue
        exs.append({"transcript": c.transcript, "diag": diag, "label": int(c.label), "p1": p1})
    return exs, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-bundle", required=True,
                    help="KorCCViD training bundle (e.g. native_trainset.jsonl) — KorMMP forbidden (frozen). Comma-separated for multiple")
    ap.add_argument("--synth-bundle", default="",
                    help="Add synthetic (seed99) hardset bundle (prosody-transfer complete). Optional — reinforce hard edges")
    ap.add_argument("--ckpt", default="artifacts/models/channel_extractors.pt")
    ap.add_argument("--head", default="artifacts/models/miltl_head.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tau-low", type=float, default=0.40)
    ap.add_argument("--tau-high", type=float, default=0.75)
    ap.add_argument("--blend", type=float, default=1.0)
    ap.add_argument("--anchor", type=int, default=80)
    ap.add_argument("--band", default="escalate", choices=["escalate", "all", "confident"],
                    help="Training target band (escalate=true Gate-2 load)")
    ap.add_argument("--gate2-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default="artifacts/models/gate2_adapter")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--fair-audio", action="store_true",
                    help="extract diag with the same channel equalization as evaluation (bench --fair-audio) — train/eval distribution alignment (docs/BENCHMARK.md)")
    ap.add_argument("--dry-run", action="store_true", help="Skip SLM loading — examples/band stats only (DGX-free)")
    args = ap.parse_args()

    bundles = args.train_bundle.split(",") + ([args.synth_bundle] if args.synth_bundle else [])
    exs, stats = build_examples(bundles, args.ckpt, args.head, args.device,
                                args.tau_low, args.tau_high, args.blend, args.anchor, args.band,
                                codec_equalize=args.fair_audio)
    pos = sum(e["label"] for e in exs)
    print(f"train bundle bands {stats} · target('{args.band}') {len(exs)} (harm={pos} benign={len(exs)-pos})",
          flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out).parent / "gate2_sft_meta.json").write_text(json.dumps(
        {"train_bundle": args.train_bundle, "band": args.band, "bands": stats,
         "n": len(exs), "pos": pos, "gate2_model": args.gate2_model, "epochs": args.epochs},
        ensure_ascii=False, indent=1), encoding="utf-8")
    if args.dry_run:
        print("dry-run: model not loaded. Recording escalate examples/band stats only.", flush=True)
        return 0
    if pos in (0, len(exs)) or not exs:
        print("Cannot train (single class / empty set) — check τ or band."); return 1

    from adapters.baselines.gate2_slm import Gate2SLM
    g2 = Gate2SLM(model_name=args.gate2_model, finetune=True, epochs=args.epochs)
    g2.fit_channels(exs)
    g2.save(args.out)
    print(f"[Gate2SFT] Saved LoRA adapter → {args.out} · evaluate via gate2_cascade --gate2-adapter {args.out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
