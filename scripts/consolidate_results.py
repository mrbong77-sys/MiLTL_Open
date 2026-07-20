#!/usr/bin/env python3
"""Consolidated results sheet — AUROC/F1/ACC/SEN/SPE/PPV/NPV + DeLong significance (docs/BENCHMARK.md).

Journal-style summary table. Consumes per-call CSVs (columns: detector, label, score; optional
corpus/slice) from the multi-round benches (KorMMP hard, KorCCViD standard). For each
(corpus, group, model): AUROC, operating point by Youden's J -> ACC/SEN/SPE/PPV/NPV. A reference
model (default MiLTL) gets DeLong's paired-ROC test vs every other model; '*' marks p<0.05.

  python scripts/consolidate_results.py \
     --sheets 'artifacts/rounds/sheet_kormmp_*.csv:KorMMP,artifacts/rounds/sheet_korccvid_*.csv:KorCCViD' \
     --reference MiLTL-Channel --out artifacts/rounds/consolidated_results.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


# ---- AUROC + DeLong (paired ROC, same samples) ----
def _auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); r = ((cs - cnt + cs + 1) / 2.0)[inv]
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    T2 = np.empty(N); T2[J] = T
    return T2


def delong_p(y, s1, s2):
    """DeLong 2-sided p-value for AUC(s1) vs AUC(s2) on the same labels y. Correlated ROC."""
    y = np.asarray(y, int)
    if y.sum() in (0, len(y)):
        return float("nan")
    order = np.argsort(-y)                                   # positives first
    m = int(y.sum()); n = len(y) - m
    preds = np.vstack([np.asarray(s1, float), np.asarray(s2, float)])[:, order]
    pos = preds[:, :m]; neg = preds[:, m:]
    tx = np.vstack([_midrank(pos[r]) for r in range(2)])
    ty = np.vstack([_midrank(neg[r]) for r in range(2)])
    tz = np.vstack([_midrank(preds[r]) for r in range(2)])
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return 1.0
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    z = (aucs[0] - aucs[1]) / math.sqrt(var)
    return math.erfc(abs(z) / math.sqrt(2))                 # 2-sided normal


# ---- operating point (Youden's J) + confusion metrics ----
def _youden(s, y):
    best, bt = -1, 0.5
    for t in np.unique(s):
        p = s >= t
        tp = int((p & (y == 1)).sum()); fp = int((p & (y == 0)).sum())
        tn = int((~p & (y == 0)).sum()); fn = int((~p & (y == 1)).sum())
        sen = tp / max(tp + fn, 1); spe = tn / max(tn + fp, 1)
        if sen + spe - 1 > best:
            best, bt = sen + spe - 1, float(t)
    return bt


def _metrics(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    thr = _youden(s, y); p = s >= thr
    tp = int((p & (y == 1)).sum()); fp = int((p & (y == 0)).sum())
    tn = int((~p & (y == 0)).sum()); fn = int((~p & (y == 1)).sum())
    sen = tp / max(tp + fn, 1)                               # recall
    ppv = tp / max(tp + fp, 1)                               # precision
    return {
        "AUROC": _auroc(s, y),
        "ACC": (tp + tn) / max(tp + tn + fp + fn, 1),
        "SEN": sen,                                          # recall
        "SPE": tn / max(tn + fp, 1),                         # specificity
        "PPV": ppv,                                          # precision
        "NPV": tn / max(tn + fn, 1),
        "F1": (2 * ppv * sen / (ppv + sen)) if (ppv + sen) > 0 else 0.0,  # harmonic mean (precision, recall)
    }


def _group(name):
    n = name.lower()
    if "miltl-dual" in n or "naive-fusion" in n or "dual" in n:
        return "Naive fusion (multimodal)"
    if "miltl" in n:
        return "MiLTL (proposed)"
    if "wave" in n or "audio-only" in n:
        return "Audio-only"
    if "bllossom" in n:
        return "Legacy LLM (text)"
    return "Legacy ML/encoder (text)"


_GROUP_ORDER = ["Legacy ML/encoder (text)", "Legacy LLM (text)", "Audio-only",
                "Naive fusion (multimodal)", "MiLTL (proposed)"]


def _load(spec):
    """'glob:corpus,glob:corpus' -> {(corpus, detector): {'y':[], 's':[]}} pooled across matched CSVs."""
    data = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        pat, _, corpus = part.partition(":")
        corpus = corpus or "bench"
        files = sorted(glob.glob(pat))
        if not files:
            print(f"[WARN] no files match: {pat}", flush=True)
        for fp in files:
            for r in csv.DictReader(open(fp, encoding="utf-8")):
                det = r.get("detector", "?")
                cp = r.get("corpus", corpus)
                d = data.setdefault((cp, det), {"y": [], "s": []})
                d["y"].append(int(r["label"])); d["s"].append(float(r["score"]))
                # ablation: if a cascade row has p1 (=L1 Gate-1 alone), collect it as a separate L1 row (no extra run)
                if "Cascade" in det and str(r.get("p1", "")).strip() != "":
                    a = data.setdefault((cp, "MiLTL-L1(ablation)"), {"y": [], "s": []})
                    a["y"].append(int(r["label"])); a["s"].append(float(r["p1"]))
    return data


def run(spec, reference, out):
    data = _load(spec)
    corpora = sorted({c for c, _ in data})
    rows = []
    for cp in corpora:
        dets = [d for (c, d) in data if c == cp]
        # reference score vector for DeLong (match by substring)
        ref = next((d for d in dets if reference.lower() in d.lower()), None)
        for det in dets:
            y = np.array(data[(cp, det)]["y"]); s = np.array(data[(cp, det)]["s"])
            m = _metrics(s, y)
            p = float("nan")
            if ref and det != ref:
                ry = np.array(data[(cp, ref)]["y"]); rs = np.array(data[(cp, ref)]["s"])
                if len(ry) == len(y):                        # same samples -> DeLong
                    p = delong_p(y, s, rs)
            rows.append({"corpus": cp, "group": _group(det), "model": det, "n": len(y),
                         **{k: round(v, 3) for k, v in m.items()},
                         "p_vs_ref": (round(p, 4) if p == p else ""),
                         "sig": ("*" if (p == p and p < 0.05) else "")})
    rows.sort(key=lambda r: (r["corpus"], _GROUP_ORDER.index(r["group"])
                             if r["group"] in _GROUP_ORDER else 9, -r["AUROC"]))
    # write CSV
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cols = ["corpus", "group", "model", "n", "AUROC", "sig", "F1", "ACC", "SEN", "SPE", "PPV", "NPV", "p_vs_ref"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    # pretty print (journal style)
    print(f"\n=== Consolidated results (reference={reference}, * = p<0.05 vs reference, DeLong) ===")
    print(f"{'Corpus':<10}{'Group':<28}{'Model':<26}{'AUROC':>9}{'F1':>7}{'ACC':>7}{'SEN':>7}"
          f"{'SPE':>7}{'PPV':>7}{'NPV':>7}")
    last = None
    for r in rows:
        cp = r["corpus"] if r["corpus"] != last else ""
        last = r["corpus"]
        print(f"{cp:<10}{r['group']:<28}{r['model'][:25]:<26}"
              f"{r['AUROC']:>8.3f}{r['sig']:<1}{r['F1']:>7.3f}{r['ACC']:>7.3f}{r['SEN']:>7.3f}"
              f"{r['SPE']:>7.3f}{r['PPV']:>7.3f}{r['NPV']:>7.3f}")
    print(f"\n[sheet] {out} ({len(rows)} rows)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", required=True,
                    help="'glob:corpus,glob:corpus' per-call CSVs (detector,label,score). e.g. "
                         "'artifacts/rounds/sheet_kormmp_*.csv:KorMMP,artifacts/rounds/sheet_korccvid_*.csv:KorCCViD'")
    ap.add_argument("--reference", default="MiLTL-Channel", help="reference model (substring) for DeLong test")
    ap.add_argument("--out", default="artifacts/rounds/consolidated_results.csv")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    rc = run(args.sheets, args.reference, args.out)
    if args.push and rc == 0:
        import subprocess
        subprocess.run(["git", "add", args.out])
        subprocess.run(["git", "commit", "-q", "-m", "consolidated results sheet"])
        for _ in range(4):
            if subprocess.run(["git", "push", "origin", "main"]).returncode == 0:
                break
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
