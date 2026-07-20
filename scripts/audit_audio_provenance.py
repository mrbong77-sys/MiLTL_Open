#!/usr/bin/env python3
"""Audio-provenance leakage audit (docs/BENCHMARK.md) — benchmark fairness guard.

If audio-based detectors (Wave-Seq etc.) shortcut-learn the **recording provenance (corpus/channel)**
instead of "phishing prosody", the audio-only AUROC is spuriously inflated. This script measures the
**collinearity** between a bundle's audio_uri source pools and labels, and reports the accuracy of a
trivial classifier that predicts the label from the pool alone (= shortcut ceiling). Near 1.0 means leakage.

  python scripts/audit_audio_provenance.py --bundle artifacts/rounds/canonical/bundle_kormmp_42.jsonl
  # also cross-check detector scores (verify score separation by pool):
  python scripts/audit_audio_provenance.py --bundle <bundle> --sheet <sheet.csv> --detector Wave-Seq
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_POOL_RE = re.compile(r"data/raw/([^/]+)")            # data/raw/<pool>/... → top-level corpus pool


def pool_of(uri: str) -> str:
    if not uri:
        return "(none)"
    m = _POOL_RE.search(uri)
    return m.group(1) if m else uri.split("!", 1)[0][:32]


def _auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); r = ((cs - cnt + cs + 1) / 2.0)[inv]
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def run(bundle: str, sheet: str = "", detector: str = "") -> int:
    rows = [json.loads(l) for l in Path(bundle).read_text(encoding="utf-8").splitlines() if l.strip()]
    lab = {r["call_id"]: int(r.get("label", r.get("harm", 0))) for r in rows}
    pool = {r["call_id"]: pool_of(r.get("audio_uri") or (r.get("meta") or {}).get("audio_uri", "")) for r in rows}

    print(f"=== audio-provenance audit: {bundle} (n={len(rows)}) ===")
    grid = defaultdict(Counter)
    for cid in lab:
        grid[pool[cid]][lab[cid]] += 1
    print(f"{'pool':<24}{'harm':>6}{'benign':>8}{'purity':>8}")
    # trivial classifier: predict each pool's majority label → accuracy = shortcut ceiling
    correct = 0
    for p, c in sorted(grid.items(), key=lambda kv: -sum(kv[1].values())):
        h, b = c[1], c[0]
        maj = max(h, b); correct += maj
        print(f"{p:<24}{h:>6}{b:>8}{maj / max(h + b, 1):>8.2f}")
    shortcut = correct / max(len(lab), 1)
    print(f"\n[shortcut] pool→label trivial-classifier accuracy = {shortcut:.3f}")
    n_pools_pure = sum(1 for c in grid.values() if c[0] == 0 or c[1] == 0)
    verdict = ("★LEAK: pool provenance nearly determines label — audio-only performance suspected to be a provenance shortcut (not prosody discrimination)"
               if shortcut >= 0.90 else
               "minor: some pool imbalance" if shortcut >= 0.75 else
               "OK: sufficient pool-label crossing (low provenance shortcut)")
    print(f"[verdict] {verdict}  (pure pools {n_pools_pure}/{len(grid)})")

    if sheet and detector:
        det_rows = [r for r in csv.DictReader(open(sheet, encoding="utf-8"))
                    if r.get("detector", "").startswith(detector)]
        sc = {r["call_id"]: float(r["score"]) for r in det_rows}
        common = [c for c in sc if c in pool]
        y = [lab[c] for c in common]; s = [sc[c] for c in common]
        print(f"\n=== detector '{detector}' vs pool (n={len(common)}) ===")
        print(f"  AUROC(score, label) = {_auroc(s, y):.3f}")
        for p in sorted({pool[c] for c in common}):
            ss = [sc[c] for c in common if pool[c] == p]
            print(f"  pool={p:<22} n={len(ss):>3} score mean={np.mean(ss):.3f} "
                  f"[{min(ss):.3f},{max(ss):.3f}]")
        print("  -> if per-pool score means separate sharply (combined with pure pools), that evidences score=provenance discrimination.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="bundle jsonl (call_id, label, audio_uri)")
    ap.add_argument("--sheet", default="", help="per-call CSV (detector, call_id, label, score) — score cross-check (optional)")
    ap.add_argument("--detector", default="", help="detector name prefix to cross-check (e.g. Wave-Seq)")
    args = ap.parse_args()
    return run(args.bundle, args.sheet, args.detector)


if __name__ == "__main__":
    raise SystemExit(main())
