#!/usr/bin/env python3
"""Canonical benchmark — KorCCViD + KorMMP, all detectors (incl MiLTL L1→L2 cascade), N random rounds.

Single orchestrator for producing the canonical results (simple):
  1) Generate N random seeds → record to seeds.json (random yet reproducible; reproduce via --seeds).
  2) For each corpus x seed, one 100-call bundle → score **legacy detectors + MiLTL-Cascade (the finished
     system, seamless L1→L2) in a single bench pass**.
     MiLTL is computed once as the single cascade detector (no duplication). KorCCViD excludes the
     audio-based detectors (wave/dual).
     KorMMP canonical = real FSS prosody + channel equalization (telephone band + mu-law, fair to audio
     detectors). Real FSS pool = --kormmp-inventory.
  3) Per-seed sheet (all detectors, all channel columns).  4) consolidate → journal table
     (AUROC/ACC/SEN/SPE/PPV/NPV + DeLong).

  python scripts/canonical_bench.py --rounds 5 --total 100 --device cuda \
     --ckpt artifacts/models/channel_extractors.pt --head artifacts/models/miltl_head.pt \
     --gate2-adapter artifacts/models/gate2_adapter --push
  # reproduce: --seeds 12345,67890,...
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
PY = sys.executable


def sh(cmd, fatal=True):
    """Run a subprocess. With fatal=False, return the rc instead of dying on failure (preserves partial results and push)."""
    cmd = [str(c) for c in cmd]
    print("  $ " + " ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    if rc and fatal:
        raise SystemExit(f"failed: {cmd[1]}")
    return rc


def _bench(bundle, sheet, detectors, args, codec_equalize=False):
    """Score legacy + MiLTL-Cascade on the same bundle in one pass (fair). cascade attaches the L2 SLM (+adapter)."""
    cmd = [PY, "scripts/bench_hard_slices.py", "--bundle", bundle, "--detectors", detectors,
           "--channels", "calib", "--device", args.device, "--out-csv", sheet,
           "--ckpt", args.ckpt, "--gate2-model", args.gate2_model]
    if args.head:
        cmd += ["--head", args.head]
    if args.gate2_adapter:
        cmd += ["--gate2-adapter", args.gate2_adapter]
    if codec_equalize:
        cmd += ["--codec-equalize"]                          # docs/BENCHMARK.md: channel equalization for audio detectors (common to both conditions)
    sh(cmd, fatal=False)                                     # even if one round fails, continue to next round / consolidate / push


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--total", type=int, default=100, help="calls per round")
    ap.add_argument("--seeds", default="", help="comma seeds to reuse (reproduce). Empty=random-generate")
    ap.add_argument("--ckpt", default="artifacts/models/channel_extractors.pt")
    ap.add_argument("--head", default="artifacts/models/miltl_head.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--korccvid-pool", default="artifacts/frozen/korccvid/test_pool.jsonl")
    ap.add_argument("--kormmp-inventory", default="artifacts/manifest/kormmp_real_full.jsonl",
                    help="KorMMP real-case pool (real FSS harm + real benign, transcripts inline). "
                         "Default=materialize_kormmp output (143 harm). "
                         "To widen, adjust materialize_kormmp --len-lo/-hi and regenerate")
    ap.add_argument("--kormmp-detectors", default="lexical,hf,cnn_bilstm,tree,wave,dual,bllossom,cascade")
    ap.add_argument("--korccvid-detectors", default="lexical,hf,cnn_bilstm,tree,bllossom,cascade",
                    help="KorCCViD is transcript-only → audio detectors (wave/dual) excluded. cascade=finished MiLTL system")
    ap.add_argument("--gate2-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--gate2-adapter", default="", help="L2 LoRA adapter (gate2_sft). Empty = zero-shot")
    ap.add_argument("--synth-nper", type=int, default=200)
    ap.add_argument("--per-slice", type=int, default=100)
    ap.add_argument("--out-dir", default="artifacts/rounds/canonical")
    ap.add_argument("--reference", default="MiLTL-Cascade", help="consolidate DeLong reference")
    ap.add_argument("--keep-stale", action="store_true",
                    help="keep previous rounds' sheets/bundles (default=remove to avoid consolidate contamination)")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    od = Path(args.out_dir); od.mkdir(parents=True, exist_ok=True)
    # Canonical integrity: stale sheets/bundles from previous rounds (different seeds) would get pooled
    # by consolidate → n mismatch / contamination. Remove previous artifacts at the start of a new run
    # (reproducibility via seeds.json). Use --keep-stale to retain.
    if not args.keep_stale:
        for pat in ("sheet_*.csv", "*.l2ledger.jsonl", "bundle_*.jsonl", "synth_*.jsonl", "consolidated_results.csv"):
            for f in od.glob(pat):
                f.unlink()
        print(f"[canonical] cleared stale artifacts in {od} (fresh run; use --keep-stale to retain)", flush=True)
    # Generate and record random seeds (reproducible). Both corpora and all detectors reuse the same
    # bundle for the same seed.
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    else:
        rng = random.SystemRandom()
        seeds = [rng.randrange(1, 1_000_000) for _ in range(args.rounds)]
    (od / "seeds.json").write_text(json.dumps({"seeds": seeds, "total": args.total}, indent=1),
                                   encoding="utf-8")
    print(f"[canonical] rounds={len(seeds)} seeds={seeds} (recorded -> {od}/seeds.json)", flush=True)

    for sd in seeds:                                         # KorCCViD (transcript-only standard)
        print(f"\n===== KorCCViD · seed {sd} =====", flush=True)
        bundle = od / f"bundle_korccvid_{sd}.jsonl"; sheet = od / f"sheet_korccvid_{sd}.csv"
        sh([PY, "scripts/materialize_korccvid.py", "--in", args.korccvid_pool,
            "--total", args.total, "--seed", sd, "--out", bundle], fatal=False)
        _bench(bundle, sheet, args.korccvid_detectors, args)

    # KorMMP canonical = real FSS prosody + channel equalization (docs/BENCHMARK.md). Single condition (no suffix tag).
    for sd in seeds:                                         # KorMMP (hard, lexically decorrelated, real FSS audio)
        print(f"\n===== KorMMP · seed {sd} =====", flush=True)
        syn = od / f"synth_{sd}.jsonl"; bundle = od / f"bundle_kormmp_{sd}.jsonl"
        sheet = od / f"sheet_kormmp_{sd}.csv"
        sh([PY, "scripts/synth_edgecases.py", "--n-per", args.synth_nper, "--seed", sd, "--out", syn], fatal=False)
        sh([PY, "scripts/compose_hard_kormmp.py", "--inventory", args.kormmp_inventory,
            "--per-slice", args.per_slice, "--seed", sd, "--synth", syn, "--total", args.total,
            "--cold-benign", "--out", bundle], fatal=False)
        _bench(bundle, sheet, args.kormmp_detectors, args, codec_equalize=True)  # channel equalization (fair to audio detectors)

    print(f"\n===== consolidate (journal table) =====", flush=True)
    sh([PY, "scripts/consolidate_results.py", "--sheets",
        f"{od}/sheet_korccvid_*.csv:KorCCViD,{od}/sheet_kormmp_*.csv:KorMMP",
        "--reference", args.reference, "--out", od / "consolidated_results.csv"], fatal=False)
    print(f"\n[canonical] done -> {od} (sheets, consolidated_results.csv, seeds.json)", flush=True)

    if args.push:
        for _ in range(4):
            subprocess.run(["git", "add", str(od)])
            subprocess.run(["git", "commit", "-q", "-m", f"canonical bench (rounds={len(seeds)}, seeds recorded)"])
            if subprocess.run(["git", "push", "origin", "main"]).returncode == 0:
                print("[canonical] auto-push done", flush=True); break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
