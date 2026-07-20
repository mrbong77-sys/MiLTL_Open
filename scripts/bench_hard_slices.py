#!/usr/bin/env python3
"""Hard KorMMP slice bench (docs/BENCHMARK.md G-C) — prove legacy collapse vs MiLTL survival.

Key metrics:
  · overall AUROC (all 202 calls)
  · **decorrelated AUROC** = hard slices only (hard-harm* vs hard-benign*, lexically decorrelated)
    → the thesis is legacy ~0.5 while MiLTL holds.
  · per-slice mean score (threshold-free) — harm slices must be high, benign slices low.
  · hard-slice recall(harm)/FPR(benign) @ threshold.

Detectors: lexical (lexical proxy) · frozen legacy (hf/cnn_bilstm/tree, trained on KorCCViD, no
retraining) · miltl (channel bottleneck).
If frozen weights are missing the detector is skipped automatically (needs a separate one-time training).

  # all legacy (frozen) + MiLTL true capability (DGX):
  python scripts/bench_hard_slices.py --bundle artifacts/manifest/kormmp_h100_42.jsonl \
     --detectors lexical,hf,cnn_bilstm,tree,miltl --channels calib \
     --head artifacts/models/miltl_head.pt --device cuda --out-csv artifacts/rounds/t.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.train_channel_extractors import _auroc
from scripts.lexical_shortcut import _lex_scores

_HARD_HARM = ("hard-harm-asr", "hard-harm-natural", "synth-hard-harm")
_HARD_BENIGN = ("hard-benign", "synth-hard-benign")
_B3_MODEL = "Herry443/Llama-8B-KNUT-ref-voice_size500_cot1_cri1_hint1_retrain"  # public finetune (quick_bench_ko)


def _f1_at(scores, labels, thr):
    p = (np.asarray(scores) >= thr).astype(int); y = np.asarray(labels).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return (2 * prec * rec / (prec + rec)) if prec + rec else 0.0


def _best_thr(scores, labels):
    best, bt = 0.0, 0.5
    for t in np.unique(scores):
        f = _f1_at(scores, labels, t)
        if f > best:
            best, bt = f, float(t)
    return bt, best


def _lexical_scorer():
    def s(call):
        return _lex_scores(getattr(call, "transcript", "") or "")["harm_lex"]
    return s


def _miltl_detector(ckpt, head, device, channels, xm_weight, blend, anchor):
    from adapters.baselines.native_channel import ChannelBottleneckDetector
    return ChannelBottleneckDetector(ckpt=ckpt, head=head, device=device, channels=channels,
                                     xm_weight=xm_weight, blend_analytic=blend, anchor_words=anchor)


# Frozen legacy baseline registry (trained on KorCCViD, retraining forbidden). name→(module, class, frozen_dir, weight_files).
_FROZEN = {
    "hf": ("adapters.baselines.hf_encoder", "HFEncoderDetector",
           "artifacts/frozen/korccvid/hf_encoder", ("model.safetensors", "pytorch_model.bin")),
    "cnn_bilstm": ("adapters.baselines.cnn_bilstm_fasttext", "CnnBiLstmFastTextDetector",
                   "artifacts/frozen/korccvid/cnn_bilstm", ("model.pt",)),
    "tree": ("adapters.baselines.tree_ensemble", "CatBoostDetector",
             "artifacts/frozen/korccvid/tree", ("tree.pkl",)),
}


def _modality(name):
    """Detector name → access modality (fairness column): the results sheet alone should show it is not 'text-only lost because it was text-only'."""
    if "audio-only" in name or "Wave" in name:
        return "audio"
    if "MiLTL" in name or "Dual" in name or "fusion" in name:
        return "multimodal"
    return "text"                                            # lexical, hf, cnn, tree, Bllossom = text-only


def _load_frozen(name):
    """Load a frozen legacy baseline (no retraining). Returns None (skip) if weights are missing."""
    import importlib
    mod, cls, path, weights = _FROZEN[name]
    if not any((Path(path) / w).is_file() for w in weights):
        print(f"  [WARN] [{name}] frozen weights missing ({path}) -> skip (needs one-time training)", flush=True)
        return None
    det = getattr(importlib.import_module(mod), cls)()
    det.load(path)
    print(f"  [{name}] loaded frozen {path} (no retraining)", flush=True)
    return det


def run(bundle, detectors, ckpt, head, device, channels, xm_weight, out_csv="", push=False, blend=1.0, anchor=80,
        gate2_model="Qwen/Qwen2.5-1.5B-Instruct", gate2_adapter="", codec_equalize=False):
    rows = [json.loads(l) for l in Path(bundle).read_text(encoding="utf-8").splitlines() if l.strip()]
    calls = [SimpleNamespace(transcript=r.get("transcript", ""), audio_uri=r.get("audio_uri"),
                             label=int(r.get("label", 0)), slice=r.get("slice", "?")) for r in rows]
    scenarios = [r.get("scenario", r.get("meta", {}).get("scenario", "")) for r in rows]
    nwords = [r.get("meta", {}).get("n_words", len(r.get("transcript", "").split())) for r in rows]
    density = [r.get("meta", {}).get("density", 0.0) for r in rows]
    csv_records = []
    y = np.array([c.label for c in calls])
    sl = np.array([c.slice for c in calls])
    hard_mask = np.array([s in _HARD_HARM or s in _HARD_BENIGN for s in sl])

    scorers = {}                                            # name -> (score_fn, detector_or_None)
    if "lexical" in detectors:
        scorers["lexical(text-proxy)"] = (_lexical_scorer(), None)
    for nm in detectors:                                    # frozen legacy (hf/cnn_bilstm/tree)
        if nm in _FROZEN:
            d = _load_frozen(nm)
            if d is not None:
                scorers[f"{nm}(frozen)"] = (d.score, None)
    if "bllossom" in detectors:                             # B3: public finetuned Bllossom (Herry443, inference-only, no retraining)
        from adapters.baselines.bllossom_repro import BllossomReproDetector
        print(f"  [bllossom] loading public finetuned {_B3_MODEL} (inference-only, no retraining)...", flush=True)
        b = BllossomReproDetector(model_name=_B3_MODEL)
        b.fit([])                                           # only loads the public weights
        scorers["Bllossom-B3(public-finetuned)"] = (b.score, None)
    if "wave" in detectors:                                 # audio-only (fairness): prosody CNN, no text access
        try:
            from adapters.baselines.wave_seq import WaveSeqDetector
            w = WaveSeqDetector(codec_equalize=codec_equalize)
            scorers["Wave-Seq(audio-only)"] = (w.score, None)
            print("  [wave] loaded audio-only frozen (prosody CNN, no text access)", flush=True)
        except Exception as e:                              # noqa: BLE001
            print(f"  [WARN] [wave] load failed, skip: {e}", flush=True)
    if "dual" in detectors:                                 # naive fusion (fairness): text⊕wave max-OR, no XM
        try:
            from adapters.baselines.wave_seq import MiLTLDualDetector
            du = MiLTLDualDetector(codec_equalize=codec_equalize)
            scorers["MiLTL-Dual(naive-fusion text+wave)"] = (du.score, None)
            print("  [dual] loaded naive-fusion frozen (text+wave OR, no neutrosophic XM)", flush=True)
        except Exception as e:                              # noqa: BLE001
            print(f"  [WARN] [dual] load failed, skip: {e}", flush=True)
    if "miltl" in detectors:                                # L1 alone (Gate-1) — optional (for comparison)
        det = _miltl_detector(ckpt, head, device, channels, xm_weight, blend, anchor)
        scorers[f"MiLTL-Channel({channels})"] = (det.score, det)
    if "cascade" in detectors:                              # ★finished MiLTL system = seamless L1→L2 cascade (single detector)
        from adapters.baselines.native_channel import MiLTLCascadeDetector
        cas = MiLTLCascadeDetector(ckpt=ckpt, head=head, device=device,
                                   gate2_model=gate2_model, gate2_adapter=gate2_adapter, anchor_words=anchor,
                                   codec_equalize=codec_equalize)
        scorers["MiLTL-Cascade"] = (cas.score, cas)
        print(f"  [cascade] MiLTL L1→L2 seamless (Gate-2={gate2_model}"
              f"{'+LoRA' if gate2_adapter else ' zero-shot'})", flush=True)

    print(f"\n=== Hard KorMMP slice bench (n={len(y)}, harm={int(y.sum())}) ===", flush=True)
    latencies = {}                                          # name -> per-call ms (measured edge budget)
    fairness = []                                            # (name, modality, overall, decorrelated) fairness comparison table
    ledger_records = []                                       # cascade L2 judgment ledger (prompt, summary, rationale, XAI) — paper post-hoc analysis & reproduction
    for name, (sc, det) in scorers.items():
        import time
        scores = np.zeros(len(calls)); diags = []; per_ms = []
        for i, c in enumerate(calls):
            t0 = time.perf_counter()
            scores[i] = float(sc(c))
            per_ms.append((time.perf_counter() - t0) * 1000.0)
            diags.append(dict(getattr(det, "last_diag", {})) if det is not None else {})
        pm = np.array(per_ms)
        latencies[name] = (float(pm.mean()), float(np.percentile(pm, 50)), float(np.percentile(pm, 95)))
        au_all = _auroc(scores, y)
        au_hard = _auroc(scores[hard_mask], y[hard_mask])
        thr, f1 = _best_thr(scores, y)
        _hh = hard_mask & (y == 1); _hb = hard_mask & (y == 0)
        _fpr = float(((scores >= thr) & _hb).sum() / max(_hb.sum(), 1))
        fairness.append((name, _modality(name), au_all, au_hard, f1, _fpr))
        print(f"\n  [{name}]  (modality={_modality(name)})", flush=True)
        print(f"    overall AUROC = {au_all:.3f} · best-F1 = {f1:.3f}", flush=True)
        _verdict = ("n/a(standard, no hard slices)" if au_hard != au_hard        # NaN = standard bench (no hard slices)
                    else 'SURVIVES' if au_hard >= 0.7 else ('marginal' if au_hard >= 0.6 else 'COLLAPSE(~random)'))
        print(f"    **decorrelated AUROC (hard slices) = {au_hard:.3f}**  {_verdict}", flush=True)
        # per-slice mean score (threshold-free) + channel diagnostics (T/I/F/E/XM)
        for s in ("easy-harm", *_HARD_HARM, "easy-benign", *_HARD_BENIGN):
            mask = sl == s
            if not mask.any():
                continue
            line = f"      {s:<20} n={int(mask.sum()):<3} mean_score={scores[mask].mean():.3f}"
            if det is not None:
                ch = {k: np.mean([d.get(k, 0.0) for d, mm in zip(diags, mask) if mm])
                      for k in ("T", "I", "F", "E", "XM")}
                line += (f"  · T={ch['T']:.2f} I={ch['I']:.2f} F={ch['F']:.2f} "
                         f"E={ch['E']:.2f} XM={ch['XM']:.3f}")
            print(line, flush=True)
        # hard-slice recall/FPR @ best-thr
        hh = hard_mask & (y == 1); hb = hard_mask & (y == 0)
        rec = float(((scores >= thr) & hh).sum() / max(hh.sum(), 1))
        fpr = float(((scores >= thr) & hb).sum() / max(hb.sum(), 1))
        print(f"    hard slices @thr={thr:.3f}: recall(hard-harm)={rec:.3f} / FPR(hard-benign)={fpr:.3f}", flush=True)
        lm = latencies[name]
        print(f"    latency (edge budget): mean={lm[0]:.1f}ms / p50={lm[1]:.1f}ms / p95={lm[2]:.2f}ms per call", flush=True)
        if det is not None:
            _xm_distribution(diags, sl)
            _scenario_breakdown(scores, sl, np.array(scenarios), y, thr)
            _qualitative(calls, scores, diags, y, sl, thr)
        # per-call CSV records (for quantitative and qualitative interpretation)
        for i, c in enumerate(calls):
            d = diags[i]
            csv_records.append({"detector": name, "modality": _modality(name),
                                "call_id": getattr(c, "call_id", rows[i].get("call_id", i)),
                                "slice": c.slice, "scenario": scenarios[i], "label": c.label,
                                "transcript": getattr(c, "transcript", "") or "",  # the judged transcript (txt/ASR) — for intuitive per-call interpretation
                                "score": round(float(scores[i]), 4),
                                "pred": int(scores[i] >= thr),
                                "outcome": _outcome(c.label, scores[i] >= thr),
                                "T": round(d.get("T", 0), 3), "I": round(d.get("I", 0), 3),
                                "F": round(d.get("F", 0), 3), "E": round(d.get("E", 0), 3),
                                "XM": round(d.get("XM", 0), 3), "V": round(d.get("V", 0), 3),
                                "cold": round(d.get("cold", 0), 3), "warmth": round(d.get("warmth", 0), 3),
                                "n_words": nwords[i], "density": density[i],
                                # cascade also logs L1 Gate-1 p1/band/decision → consolidate extracts the L1 ablation
                                "p1": (round(float(d["p1"]), 4) if "p1" in d else ""),
                                "band": d.get("band", ""), "decision": d.get("decision", "")})
            # L2 judgment ledger: for escalate, summary/prompt/P(harm); for all calls, XAI reasons/actions (post-hoc analysis, reproduction, XAI)
            if d.get("_xai") is not None or d.get("_l2") is not None:
                ledger_records.append({
                    "detector": name, "call_id": getattr(c, "call_id", rows[i].get("call_id", i)),
                    "slice": c.slice, "label": c.label, "p1": d.get("p1"), "band": d.get("band"),
                    "decision": d.get("decision"), "final_score": round(float(scores[i]), 4),
                    "channels": {k: round(d.get(k, 0), 3) for k in ("T", "I", "F", "E", "XM", "cold", "warmth")},
                    "gate2": d.get("_l2"),           # {p2, summary(MiLTL→LMM handoff), prompt(reproduction)} · null for confident bands
                    "xai": d.get("_xai")})           # {verdict, reasons[], action[], summary}
    # Fairness comparison table — the results sheet alone should show it is not "lost because it was text-only" (collapse/survival by modality)
    print(f"\n=== Fairness comparison (decorrelated by modality) ===", flush=True)
    print(f"  {'detector':<28}{'modality':>12}{'overall':>9}{'decorr':>9}{'F1':>7}{'FPR':>7}   verdict", flush=True)
    order = {"text": 0, "audio": 1, "multimodal": 2}
    for name, mod, oa, da, f1, fp in sorted(fairness, key=lambda r: (order.get(r[1], 9), -r[3])):
        v = "n/a(std)" if da != da else ("SURVIVES" if da >= 0.7 else ("marginal" if da >= 0.6 else "COLLAPSE"))
        print(f"  {name:<28}{mod:>12}{oa:>9.3f}{da:>9.3f}{f1:>7.3f}{fp:>7.3f}   {v}", flush=True)
    print("  Note: audio-only mis-flags cold normal calls (FPR up, uncalibrated); naive-fusion collapses from text contamination -> MiLTL edge = XM (cross-modal calibration).", flush=True)
    # Edge budget summary (latency vs effectiveness) — evidence for Gate-1 (always-on edge) vs Gate-2 (server LLM)
    print(f"\n=== Edge budget summary (latency vs effectiveness) ===", flush=True)
    print(f"  {'detector':<26}{'mean ms':>10}{'p95 ms':>10}   verdict", flush=True)
    for name, lm in latencies.items():
        verdict = "edge Gate-1 OK" if lm[0] < 50 else ("server/Gate-2 only" if lm[0] > 1000 else "borderline")
        print(f"  {name:<26}{lm[0]:>10.1f}{lm[2]:>10.1f}   {verdict}", flush=True)
    if out_csv and csv_records:
        import csv
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_records[0].keys()))
            w.writeheader(); w.writerows(csv_records)
        print(f"\n[results] per-call CSV {len(csv_records)} rows -> {out_csv}", flush=True)
        if ledger_records:                                   # L2 judgment ledger (prompt, summary, rationale) — paper post-hoc analysis & reproduction
            lp = str(out_csv).rsplit(".csv", 1)[0] + ".l2ledger.jsonl"
            with open(lp, "w", encoding="utf-8") as f:
                for r in ledger_records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[results] L2 ledger {len(ledger_records)} rows -> {lp}", flush=True)
            if push:
                _push_results(lp)
        if push:
            _push_results(out_csv)
    return 0


def _push_results(path):
    """Auto commit and push the results sheet (DGX)."""
    import subprocess
    try:
        subprocess.run(["git", "add", path], check=True)
        subprocess.run(["git", "commit", "-q", "-m", f"bench results {Path(path).name}"], check=True)
        for _ in range(4):
            if subprocess.run(["git", "push", "origin", "main"]).returncode == 0:
                print(f"[results] auto-push done {path}", flush=True); return
        print("[results] [WARN] push failed (check manually)", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"[results] [WARN] commit skipped ({e})", flush=True)


def _outcome(label, pred):
    return {(1, True): "TP", (1, False): "FN", (0, True): "FP", (0, False): "TN"}[(int(label), bool(pred))]


def _scenario_breakdown(scores, sl, scen, y, thr):
    """Per-scenario mean_score and detection/false-alarm rates — qualitative analysis (which scenarios are weak)."""
    print(f"\n    [by scenario] (synthetic slices)", flush=True)
    for s in ("synth-hard-harm", "synth-hard-benign"):
        mask = sl == s
        scs = sorted(set(scen[mask]) - {""})
        for sc in scs:
            m2 = mask & (scen == sc)
            if m2.sum() == 0:
                continue
            det_rate = float((scores[m2] >= thr).mean())
            tag = "recall" if y[m2][0] == 1 else "FPR"
            print(f"      {s:<18} {sc:<8} n={int(m2.sum()):<3} mean={scores[m2].mean():.3f} {tag}={det_rate:.2f}", flush=True)


def _qualitative(calls, scores, diags, y, sl, thr):
    """Worst top-3 errors (FN and FP) — with channel values, for failure-mode diagnosis."""
    hard = np.array([s in _HARD_HARM or s in _HARD_BENIGN for s in sl])
    fn = [i for i in np.argsort(scores) if hard[i] and y[i] == 1 and scores[i] < thr][:3]     # lowest-scoring harm
    fp = [i for i in np.argsort(-scores) if hard[i] and y[i] == 0 and scores[i] >= thr][:3]   # highest-scoring benign
    print(f"\n    [qualitative failure modes]", flush=True)
    for tag, idxs in (("FN(missed phishing)", fn), ("FP(benign false-alarm)", fp)):
        for i in idxs:
            d = diags[i]
            print(f"      {tag} {sl[i]:<18} score={scores[i]:.3f} XM={d.get('XM',0):.3f} "
                  f"cold={d.get('cold',0):.2f} T={d.get('T',0):.2f} F={d.get('F',0):.2f}", flush=True)


def _xm_distribution(diags, sl):
    """Per-call XM dispersion diagnosis — where in the distribution the signal lives + cold correlation (root-causing prosody transplantation)."""
    def col(k, mask):
        return np.array([d.get(k, 0.0) for d, m in zip(diags, mask) if m])
    ben_mask = np.array([s in _HARD_BENIGN for s in sl])
    xm_ben = col("XM", ben_mask)
    ben_p90 = float(np.percentile(xm_ben, 90)) if len(xm_ben) else 0.0
    print(f"\n    [per-call XM distribution] hard-benign XM p90={ben_p90:.3f} (detection-threshold proxy)", flush=True)
    print(f"      {'slice':<20}{'XM p10':>8}{'p50':>7}{'p90':>7}{'cold p50':>9}{'XM>benp90 %':>12}", flush=True)
    for s in ("synth-hard-harm", "hard-harm-asr", "easy-harm", "hard-benign", "synth-hard-benign"):
        mask = sl == s
        if not mask.any():
            continue
        xm = col("XM", mask); cold = col("cold", mask)
        frac = float((xm > ben_p90).mean() * 100) if len(xm) else 0.0
        print(f"      {s:<20}{np.percentile(xm,10):>8.3f}{np.percentile(xm,50):>7.3f}"
              f"{np.percentile(xm,90):>7.3f}{np.percentile(cold,50):>9.3f}{frac:>11.0f}%", flush=True)
    # XM~cold correlation on synth-hard-harm (does prosody transplantation raise XM only when the audio is cold?)
    m = sl == "synth-hard-harm"
    if m.any():
        xm = col("XM", m); cold = col("cold", m)
        if xm.std() > 1e-6 and cold.std() > 1e-6:
            r = float(np.corrcoef(xm, cold)[0, 1])
            print(f"      synth-hard-harm XM~cold corr r={r:.2f}  "
                  f"{'-> colder audio -> higher XM (hypothesis confirmed)' if r > 0.3 else '-> cold-independent (XM from other factor)'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--detectors", default="lexical")
    ap.add_argument("--ckpt", default="artifacts/models/channel_extractors.pt")
    ap.add_argument("--head", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--channels", choices=["calib", "extractor"], default="calib",
                    help="MiLTL channel source: calib=analytic (text768-free, recommended) / extractor=learned")
    ap.add_argument("--xm-weight", type=float, default=3.0, help="XM (cross-modal novelty) main-term weight")
    ap.add_argument("--blend", type=float, default=1.0,
                    help="analytic weight (1.0=pure analytic E-2T+I+XM=canonical, 0=pure head). Only mixed when head loaded")
    ap.add_argument("--anchor", type=int, default=80, help="anchor word count (below=undecidable 0.0). lowered 210->80 (recall)")
    ap.add_argument("--out-csv", default="", help="per-call results CSV path")
    ap.add_argument("--push", action="store_true", help="auto commit/push results (DGX)")
    ap.add_argument("--gate2-model", default="Qwen/Qwen2.5-1.5B-Instruct", help="cascade L2 SLM backbone (canonical 1.5B)")
    ap.add_argument("--gate2-adapter", default="", help="cascade L2 LoRA adapter (gate2_sft). Empty=zero-shot")
    ap.add_argument("--codec-equalize", action="store_true",
                    help="fair audio equalization (docs/BENCHMARK.md): apply telephone band + mu-law uniformly to all audio detectors")
    ap.add_argument("--fair-audio", action="store_true", help="alias for --codec-equalize (backward compat)")
    args = ap.parse_args()
    return run(args.bundle, args.detectors.split(","), args.ckpt, args.head, args.device,
               args.channels, args.xm_weight, out_csv=args.out_csv, push=args.push, blend=args.blend,
               anchor=args.anchor, gate2_model=args.gate2_model, gate2_adapter=args.gate2_adapter,
               codec_equalize=(args.codec_equalize or args.fair_audio))


if __name__ == "__main__":
    raise SystemExit(main())
