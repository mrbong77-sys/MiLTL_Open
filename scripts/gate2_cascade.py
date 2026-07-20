#!/usr/bin/env python3
"""MiLTL cascade — Gate-1 (channel bottleneck) → band routing → Gate-2 (lightweight SLM, escalate only) (docs/BENCHMARK.md).

Gate-1 = ChannelBottleneckDetector (canonical E−2T+I+XM). Band split on p1:
  p1≤τ_low → terminate as benign · p1≥τ_high → early harm alert · in between → **escalate** (call Gate-2).
Gate-2 = Gate2SLM.score_channels(transcript + **full XM channel summary**). Called only on escalate → minimal latency/cost.
Canonical comparison = cascade precision gain vs Gate-1 alone (escalate treated as harm).

  # DGX (SLM loaded, auto-push ledger and summary):
  python scripts/gate2_cascade.py --bundle artifacts/rounds/hardX_42.jsonl \
     --ckpt artifacts/models/channel_extractors.pt --head artifacts/models/miltl_head.pt \
     --gate2-model Qwen/Qwen2.5-0.5B-Instruct --device cuda \
     --out-ledger artifacts/rounds/gate2_ledger_42.jsonl --out-csv artifacts/rounds/gate2_summary_42.csv --push
  # DGX-free validation (SLM not loaded; band distribution + oracle ceiling + summary/prompt recorded):
  python scripts/gate2_cascade.py --bundle <bundle> --dry-run --out-ledger <ledger.jsonl>

Ledger (--out-ledger) = per-call JSONL: gate1{p1, band, T/I/F/E/XM, cold, warmth} +
  gate2{summary (MiLTL-generated summary), prompt (full, for reproduction), rationale (LMM rationale), raw, decision}.
  → post-hoc analysis & reproduction: same prompt + model = same judgment. Fully preserves the
  MiLTL-logic-produced summary and the LMM's judgment rationale.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def _prf(y, pred):
    y = np.asarray(y); pred = np.asarray(pred)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    rec = tp / max(tp + fn, 1); prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"recall": rec, "precision": prec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


_BANDNM = {0: "benign", 1: "escalate", 2: "harm"}


def _outcome(label, pred):
    return ("TP" if pred else "FN") if label == 1 else ("FP" if pred else "TN")


def run(bundle, ckpt, head, device, tau_low, tau_high, gate2_model, dry_run, blend, anchor,
        out_ledger, out_csv, push, gate2_adapter=""):
    rows = [json.loads(l) for l in Path(bundle).read_text(encoding="utf-8").splitlines() if l.strip()]
    calls = [SimpleNamespace(transcript=r.get("transcript", ""), audio_uri=r.get("audio_uri"),
                             label=int(r.get("label", 0)), slice=r.get("slice", "?"),
                             call_id=r.get("call_id", i)) for i, r in enumerate(rows)]
    y = np.array([c.label for c in calls])

    from adapters.baselines.native_channel import ChannelBottleneckDetector
    from adapters.baselines.gate2_slm import summarize_channels
    g1 = ChannelBottleneckDetector(ckpt=ckpt, head=head, device=device, channels="calib",
                                   blend_analytic=blend, anchor_words=anchor)
    p1 = np.zeros(len(calls)); diags = []
    for i, c in enumerate(calls):
        p1[i] = float(g1.score(c))
        diags.append(dict(getattr(g1, "last_diag", {})))

    band = np.where(p1 <= tau_low, 0, np.where(p1 >= tau_high, 2, 1))  # 0 benign · 2 harm · 1 escalate
    n_esc = int((band == 1).sum())
    print(f"=== MiLTL cascade (n={len(y)}, harm={int(y.sum())}) ===", flush=True)
    print(f"bands (tau_low={tau_low} tau_high={tau_high}): benign={int((band==0).sum())} "
          f"harm={int((band==2).sum())} escalate={n_esc} (escalation_rate={n_esc/len(y):.1%})", flush=True)
    esc_y = y[band == 1]
    if n_esc:
        print(f"  escalate band: harm={int(esc_y.sum())} benign={int((esc_y==0).sum())} "
              f"— actual load Gate-2 must resolve", flush=True)

    g1_pred = (band >= 1).astype(int)
    m1 = _prf(y, g1_pred)
    print(f"\n[Gate-1 alone] escalate treated as harm: "
          f"recall={m1['recall']:.3f} precision={m1['precision']:.3f} F1={m1['f1']:.3f} "
          f"(FP={m1['fp']})", flush=True)

    # Gate-2 judgment (escalate only). dry-run=oracle (ceiling), but summary/prompt recorded identically.
    g2 = None
    if not dry_run:
        from adapters.baselines.gate2_slm import Gate2SLM
        g2 = Gate2SLM(model_name=gate2_model, finetune=False)
        if gate2_adapter:                                   # load trained LoRA adapter (gate2_sft)
            try:
                g2.load_adapter(gate2_adapter)
                print(f"\n[Gate-2] {gate2_model} + LoRA adapter {gate2_adapter}: judging {n_esc} escalated calls...", flush=True)
            except ImportError:                             # peft not installed → zero-shot fallback (L2 records kept)
                print(f"\n[Gate-2] [WARN] LoRA adapter load failed (peft not installed) -> zero-shot fallback. "
                      f"To use the adapter, `pip install peft`.", flush=True)
                g2.fit([])
        else:
            g2.fit([])                                      # load public weights (zero-shot)
            print(f"\n[Gate-2] {gate2_model} zero-shot: judging {n_esc} escalated calls (XM injected)...", flush=True)

    final = band.copy()
    ledger = []                                             # ★ledger: full per-call MiLTL summary and LMM judgment
    esc_k = 0
    for i, c in enumerate(calls):
        d = diags[i]
        rec = {"call_id": c.call_id, "slice": c.slice, "label": int(y[i]),
               "gate1": {"p1": round(p1[i], 4), "band": _BANDNM[int(band[i])],
                         **{k: round(float(d.get(k, 0.0)), 3) for k in
                            ("T", "I", "F", "E", "XM", "cold", "warmth")}},
               "gate2": None}
        if band[i] == 1:                                    # escalate → Gate-2 (record summary, prompt, rationale)
            if dry_run:                                     # oracle: record only summary/prompt (for reproduction), decision=label
                jd = g2.judge_channels(c.transcript, d) if g2 else \
                    {"summary": summarize_channels(d), "prompt": "", "rationale": "", "raw": "",
                     "decision": None}
                decision = "harm" if y[i] == 1 else "benign"
                jd["decision"] = decision + "(oracle)"
            else:
                jd = g2.judge_channels(c.transcript, d)
                decision = jd["decision"]
                esc_k += 1
                if esc_k <= 3:
                    print(f"  esc#{esc_k} label={y[i]} XM={d.get('XM',0):.2f} -> {decision} · "
                          f"rationale: {jd['rationale'][:80]}", flush=True)
            final[i] = 2 if decision.startswith("harm") else 0
            rec["gate2"] = {"model": (gate2_model if not dry_run else "oracle"),
                            "summary": jd["summary"], "prompt": jd["prompt"],
                            "rationale": jd["rationale"], "raw": jd["raw"], "decision": decision}
        rec["final"] = _BANDNM[int(final[i])] if final[i] != 1 else "escalate"
        rec["outcome"] = _outcome(int(y[i]), int(final[i] == 2))
        # ★XAI: channel signals + decision → user-facing reasons and recommended actions (deterministic sanitize, basis for edge actions)
        from miltl.native.explain import explain_decision
        rec["explanation"] = explain_decision(d, "harm" if final[i] == 2 else "benign",
                                              c.transcript, float(p1[i]))
        ledger.append(rec)

    casc_pred = (final == 2).astype(int)
    m2 = _prf(y, casc_pred)
    tag = "oracle(ceiling)" if dry_run else f"SLM({gate2_model})"
    print(f"\n[cascade final] Gate-2={tag}: "
          f"recall={m2['recall']:.3f} precision={m2['precision']:.3f} F1={m2['f1']:.3f} "
          f"(FP={m2['fp']} -> reduced from Gate-1 {m1['fp']} by {m1['fp']-m2['fp']})", flush=True)
    print(f"\nsummary: Gate-1 F1 {m1['f1']:.3f} -> cascade F1 {m2['f1']:.3f} / "
          f"escalation {n_esc/len(y):.1%} (Gate-2 call ratio = latency/cost) / "
          f"precision {m1['precision']:.3f}->{m2['precision']:.3f}", flush=True)
    # XAI rationale samples (one risky, one safe) — preview of the edge user-facing report
    print("\n=== Decision rationale (XAI) samples ===", flush=True)
    for want in (2, 0):
        s = next((r for r in ledger if (r["final"] == "harm") == (want == 2)), None)
        if s:
            print(f"  [{s['final']}·{s['outcome']}] {s['explanation']['summary'][:200]}", flush=True)

    _write_records(ledger, m1, m2, n_esc, len(y), tag, tau_low, tau_high, out_ledger, out_csv, push)
    return 0


def _write_records(ledger, m1, m2, n_esc, n, tag, tau_low, tau_high, out_ledger, out_csv, push):
    """Save ledger (JSONL, per-call MiLTL summary, LMM rationale, decision) + summary (CSV). For post-hoc analysis and reproduction."""
    pushed = []
    if out_ledger:
        Path(out_ledger).parent.mkdir(parents=True, exist_ok=True)
        Path(out_ledger).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in ledger) + "\n", encoding="utf-8")
        print(f"[ledger] per-call decision records {len(ledger)} rows -> {out_ledger}", flush=True)
        pushed.append(out_ledger)
    if out_csv:
        import csv
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["metric", "gate1_alone", "cascade", "gate2", "escalation_rate",
                        "tau_low", "tau_high"])
            for k in ("recall", "precision", "f1", "fp"):
                w.writerow([k, round(m1[k], 4), round(m2[k], 4), tag, round(n_esc / n, 4),
                            tau_low, tau_high])
        print(f"[summary] cascade metrics -> {out_csv}", flush=True)
        pushed.append(out_csv)
    if push and pushed:
        import subprocess
        subprocess.run(["git", "add", *pushed])
        subprocess.run(["git", "commit", "-q", "-m", f"Gate-2 cascade ledger ({tag})"])
        for _ in range(4):
            if subprocess.run(["git", "push", "origin", "main"]).returncode == 0:
                print("[ledger] auto-push done", flush=True); return
        print("[ledger] [WARN] push failed (manual)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--ckpt", default="artifacts/models/channel_extractors.pt")
    ap.add_argument("--head", default="artifacts/models/miltl_head.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tau-low", type=float, default=0.40, help="below=terminate as benign")
    ap.add_argument("--tau-high", type=float, default=0.75, help="above=early harm alert")
    ap.add_argument("--blend", type=float, default=1.0)
    ap.add_argument("--anchor", type=int, default=80)
    ap.add_argument("--gate2-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--gate2-adapter", default="", help="trained LoRA adapter path (from gate2_sft). Empty=zero-shot")
    ap.add_argument("--dry-run", action="store_true", help="SLM not loaded — band distribution + oracle Gate-2 ceiling")
    ap.add_argument("--out-ledger", default="", help="per-call decision ledger JSONL (MiLTL summary, LMM rationale, decision)")
    ap.add_argument("--out-csv", default="", help="cascade metrics summary CSV")
    ap.add_argument("--push", action="store_true", help="auto commit/push ledger and summary (DGX)")
    args = ap.parse_args()
    return run(args.bundle, args.ckpt, args.head, args.device, args.tau_low, args.tau_high,
               args.gate2_model, args.dry_run, args.blend, args.anchor,
               args.out_ledger, args.out_csv, args.push, args.gate2_adapter)


if __name__ == "__main__":
    raise SystemExit(main())
