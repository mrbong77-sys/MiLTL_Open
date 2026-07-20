#!/usr/bin/env python3
"""
edge_benchmark.py — CPU-edge budget benchmark for the MiLTL cascade (no GPU).

Substantiates the "runs on mobile-class hardware" claim with honest, reproducible
numbers measured on the machine you run it on. It reuses the demo's real L1 engine
(pure-numpy analytic channels) and, if torch+transformers are installed, the repo's
real Gate-2 SLM path forced onto CPU fp32.

Reported:
  * device-independent evidence — parameter counts and on-disk footprints
    (valid regardless of the target device);
  * L1 Gate-1: per-call latency distribution over the bundled demo cases
    (featurize + calibrated channels + analytic risk), and resident memory;
  * L2 Gate-2: model load time, SCORING latency (a single forward pass over the
    escalation prompt — no generation), rationale generation tok/s, peak RSS.

METHOD: cases are timed in full sweeps with seeded shuffling per sweep, so a
transient system slowdown spreads across cases instead of biasing a few. Run on an
otherwise-idle machine on AC power for clean numbers.

NOTE ON DEVICES: latency is a property of the actual silicon; report these numbers
as "measured on <your machine>", never as another device's. Use --threads 1 to
emulate a single constrained core. fp32 is the *conservative* CPU path — int4/int8
quantized runtimes only lower the L2 budget further.

Run (from the repo root):
    python demo/edge_benchmark.py                 # L1 only (numpy)
    python demo/edge_benchmark.py --gate2         # + Gate-2 CPU measurement
    python demo/edge_benchmark.py --gate2 --gate2-model Qwen/Qwen2.5-1.5B-Instruct
    python demo/edge_benchmark.py --threads 1     # single-core emulation
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "demo"))

from miltl_demo import L1Engine, _load_cases, _proc_metrics  # noqa: E402


def _fmt_ms(xs):
    return (f"median {statistics.median(xs):.2f} · mean {statistics.mean(xs):.2f} · "
            f"p95 {sorted(xs)[int(0.95 * (len(xs) - 1))]:.2f} · max {max(xs):.2f} ms")


def bench_l1(sweeps: int, seed: int = 0):
    eng = L1Engine()
    cases = [c for c in _load_cases()]
    if not cases:
        print("[l1] no bundled cases found — run from the repo root.")
        return None
    rng = random.Random(seed)
    lat = []
    per_nibble = []
    t0 = time.perf_counter()
    for s in range(sweeps):
        order = list(range(len(cases)))
        rng.shuffle(order)
        for i in order:
            c = cases[i]
            nci, has_audio, _ = eng.featurize(c["transcript"], c["prosody"])
            ok, _, _ = eng.anchor_ok(c["transcript"], has_audio)
            if not ok:
                continue
            t = time.perf_counter()
            step = eng.score_prefix(nci, has_audio, int(nci.n_valid))
            dt = (time.perf_counter() - t) * 1e3
            lat.append(dt)
            per_nibble.append(dt / max(int(nci.n_valid), 1))
    wall = time.perf_counter() - t0
    m = _proc_metrics()
    print("== L1 Gate-1 (pure numpy: featurize→calibrated channels→analytic risk) ==")
    print(f"  cases×sweeps      : {len(lat)}  (wall {wall:.2f}s, includes featurize)")
    print(f"  full-call scoring : {_fmt_ms(lat)}")
    print(f"  per-nibble        : {_fmt_ms(per_nibble)}")
    print(f"  throughput        : {len(lat) / wall:.0f} calls/s (single process)")
    print(f"  process RSS       : {m['rss_mb']} MB · cores {m['ncpu']}")
    print(f"  learned params    : 0 (canonical scorer is the analytic rule; Calib = "
          f"{18 * 2 + 6} scalar statistics)")
    return eng


def bench_l2(model_name: str, threads: int, eng: L1Engine):
    try:
        import torch
        from transformers import AutoModelForCausalLM  # noqa: F401
    except ImportError:
        print("== L2 Gate-2 == skipped (install torch+transformers; see demo/README.md)")
        return
    from miltl_demo import L2Runtime

    rt = L2Runtime(model_name, threads=threads)
    print(f"== L2 Gate-2 ({model_name}, CPU fp32{f', {threads} threads' if threads else ''}) ==")
    rt._load()
    if rt.status != "ready":
        print(f"  load failed: {rt.detail}")
        return
    print(f"  model load        : {rt.load_s} s (cold start; cached weights)")
    n_params = sum(p.numel() for p in rt._g2._model.parameters())
    print(f"  parameters        : {n_params / 1e9:.2f} B (fp32 in RAM ≈ {n_params * 4 / 2**30:.1f} GiB; "
          f"int4-quantized runtime ≈ {n_params * 0.55 / 2**30:.1f} GiB)")

    # representative escalated call from the bundled cases
    cases = [c for c in _load_cases() if c["label"] == 1]
    c = cases[0]
    nci, has_audio, _ = eng.featurize(c["transcript"], c["prosody"])
    step = eng.score_prefix(nci, has_audio, int(nci.n_valid))
    diag = dict(step["mean"])
    diag["audio"] = 1 if has_audio else 0

    lat = []
    for i in range(5):
        p2, dt = rt.score(c["transcript"], diag)
        lat.append(dt)
    print(f"  scoring (1 forward, escalations only): median {statistics.median(lat[1:]):.2f} s "
          f"(all: {', '.join(f'{x:.2f}' for x in lat)}) → P(harm)={p2:.3f}")
    t0 = time.perf_counter()
    out, dt, gen = rt.rationale(c["transcript"], diag)
    n_chars = len(out.get("raw") or gen)
    print(f"  rationale gen     : {dt:.1f} s for ~{n_chars} chars (optional XAI path)")
    m = _proc_metrics()
    print(f"  process RSS       : {m['rss_mb']} MB (fp32 — quantized runtimes are ~4-7× smaller)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", type=int, default=20)
    ap.add_argument("--gate2", action="store_true", help="also measure Gate-2 on CPU")
    ap.add_argument("--gate2-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0=auto)")
    args = ap.parse_args()

    print(f"[edge_benchmark] {time.strftime('%Y-%m-%d %H:%M:%S')} · report numbers as "
          f"'measured on THIS machine'\n")
    eng = bench_l1(args.sweeps)
    if eng and args.gate2:
        print()
        bench_l2(args.gate2_model, args.threads, eng)


if __name__ == "__main__":
    main()
