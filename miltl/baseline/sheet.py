"""Unified result-sheet renderer (docs/BASELINES.md) — results.json (machine) + results.md (human, one table).

Rows=detectors, columns=unified metrics. Accompanied by reproducibility meta (repro/latency/skipped). Sorted by AUROC descending.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

from .detector import ResultRow

_COLS = [
    ("AUROC", "auroc"), ("pAUC@1%", "pauc_1pct"),
    ("R@FPR1%", "recall_at_fpr_1pct"), ("R@FPR.1%", "recall_at_fpr_0_1pct"),
    ("F1", "f1"), ("Acc", "accuracy"), ("Prec", "precision"), ("ECE", "ece"),
]


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.3f}"


def render_sheet(rows: Sequence[ResultRow], title: str = "베이스라인 벤치마크 결과") -> str:
    """One markdown table + footnotes. Sorted by AUROC (errors/unscored at the bottom)."""
    ranked = sorted(rows, key=lambda r: (r.error is not None, -(r.metrics.get("auroc", 0.0))))
    head = ("| 디텍터 | 계열 | 모달 | eval | " + " | ".join(c[0] for c in _COLS)
            + " | n(pos) | skip | ms/call | repro |")
    sep = "|" + "---|" * (5 + len(_COLS) + 3)
    L = [f"# {title}", "",
         "> 동일 동결 벤치마크·동일 지표(docs/18). test 재튜닝 없음(운영 임계=train 선택). "
         "skip=모달 결측 제외 수. repro: ok/partial/paper-only. "
         "eval: in-dist(동일출처 split) vs xcorpus(교차출처 일반화 — shortcut 붕괴 노출).", "",
         head, sep]
    for r in ranked:
        if r.error:
            L.append(f"| {r.name} | {r.family} | {r.modality} | {r.eval_mode} | "
                     + " | ".join("—" for _ in _COLS)
                     + f" | — | {r.skipped} | — | {r.repro} |  ⚠️ {r.error} |")
            continue
        m = r.metrics
        cells = " | ".join(_fmt(m.get(k)) for _, k in _COLS)
        L.append(f"| {r.name} | {r.family} | {r.modality} | {r.eval_mode} | {cells} | "
                 f"{r.n_test}({r.n_pos}) | {r.skipped} | {r.latency_ms:.1f} | {r.repro} |")
    notes = [r for r in ranked if r.notes]
    if notes:
        L += ["", "## 각주"]
        L += [f"- **{r.name}**: {r.notes}" for r in notes]
    return "\n".join(L) + "\n"


def render_matrix(results_by_axis, metric: str = "auroc", baseline_axis: str = None,
                  title: str = "강건성 매트릭스 (방법 × 축)") -> str:
    """Per-axis result sheets (dict {axis: [row_dict]}) → method×axis matrix + Δ (base axis − shifted axis) (docs/BENCHMARK.md).

    Headline = **max Δ** (worst robustness gap). Smaller Δ = more robust (learned intent, not vocabulary). Sorted by Δ ascending.
    row_dict is in ResultRow.to_dict() format (name·metrics). Missing/error shown as '—'.
    """
    axes = list(results_by_axis.keys())
    if not axes:
        return f"# {title}\n\n(빈 결과)\n"
    base = baseline_axis or axes[0]
    shifted = [a for a in axes if a != base]
    dets = {}                                          # name → {axis: value}
    for axis, rows in results_by_axis.items():
        for r in rows:
            v = (r.get("metrics") or {}).get(metric) if not r.get("error") else None
            dets.setdefault(r["name"], {})[axis] = v

    def maxdelta(vals):
        b = vals.get(base)
        ds = [b - vals[a] for a in shifted if b is not None and vals.get(a) is not None]
        return max(ds) if ds else None

    ranked = sorted(dets.items(), key=lambda kv: (maxdelta(kv[1]) is None, maxdelta(kv[1]) or -1))
    head = "| 방법 | " + " | ".join(axes) + " | " + " | ".join(f"Δ{a}" for a in shifted) + " | maxΔ |"
    sep = "|" + "---|" * (1 + len(axes) + len(shifted) + 1)
    L = [f"# {title}", "",
         f"> 지표={metric}. 기준축={base}. Δ=기준−이동(양수=이동 시 하락). **maxΔ 작을수록 robust**(어휘 shortcut "
         "아닌 의도 학습). docs/20 §3.", "", head, sep]
    for name, vals in ranked:
        cells = " | ".join(_fmt(vals.get(a)) for a in axes)
        deltas = []
        for a in shifted:
            b, x = vals.get(base), vals.get(a)
            deltas.append(f"{b-x:+.3f}" if (b is not None and x is not None) else "—")
        md = maxdelta(vals)
        L.append(f"| {name} | {cells} | " + " | ".join(deltas) + f" | {_fmt(md)} |")
    return "\n".join(L) + "\n"


def write_sheet(rows: List[ResultRow], out_dir: str, title: str = "베이스라인 벤치마크 결과"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps([r.to_dict() for r in rows], ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "results.md").write_text(render_sheet(rows, title), encoding="utf-8")
    return out / "results.json", out / "results.md"
