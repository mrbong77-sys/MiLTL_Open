#!/usr/bin/env python3
"""Hard KorMMP composer (docs/BENCHMARK.md) — induce legacy collapse via lexically decorrelated slices.

Proposition: once the lexicon-harm correlation is severed, legacy text models collapse while
MiLTL must survive on prosodic signals (cold, XM, trajectory).
Slices (criterion = observable lexical density, never cut on the harm label -> no circularity):
  easy-harm    : high scam-lexical-density FSS (control group; legacy also catches these)
  hard-harm    : (1) naturally low-density FSS  (2) ASR-degraded FSS (transcript only corrupted,
                 audio untouched) — legacy FN
  easy-benign  : low-density everyday conversation (trivial)
  hard-benign  : high-density benign calls (finance/authority consultations) — legacy FP
Length and modality matched, seed-deterministic. Output = inline-transcript bundle (directly runnable by the bench).

  # DGX (transcript sidecar required):
  python scripts/compose_hard_kormmp.py --inventory artifacts/manifest/case_inventory_hard.jsonl \
     --per-slice 40 --seed 42 --out artifacts/manifest/kormmp_hardX_full.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from miltl.native.channel_teacher import _SCAM, _THREAT, _DIRECTIVE, _URGENCY

_KW = list(_SCAM) + list(_THREAT) + list(_DIRECTIVE) + list(_URGENCY)


def _density(text: str) -> float:
    if not text:
        return 0.0
    nw = max(len(text.split()), 1)
    hits = sum(text.count(w) for w in _KW)
    return hits / nw * 100.0


def _asr_degrade(text: str, rng: random.Random, drop: float = 0.12, kw_corrupt: float = 0.75) -> str:
    """Simulate ASR degradation — audio is untouched, **only the transcript** is corrupted.
    Scam keywords are corrupted preferentially, plus mild dropout.

    Partially corrupting keywords (single-character edit) defeats _lex matching = removes the
    lexical shortcut. Prosodic signals are unaffected (audio preserved).
    """
    out = []
    for tok in text.split():
        is_kw = any(k in tok for k in _KW)
        if is_kw and rng.random() < kw_corrupt:
            if len(tok) >= 2:                       # delete one character (misrecognition) -> keyword match breaks
                i = rng.randrange(len(tok))
                tok = tok[:i] + tok[i + 1:]
        elif (not is_kw) and rng.random() < drop:
            continue                                # non-keyword dropout
        out.append(tok)
    return " ".join(out)


def _audio(c):
    return c.get("audio_path") or c.get("audio_uri")


def _is_harm(c):
    return c.get("class") == "harm" or int(c.get("label", 0)) == 1


def _read_transcript(case, asr_track, mirror_dir):
    if case.get("transcript"):                       # materialized bundle = inline transcript takes priority
        return case["transcript"].strip()
    from miltl.baseline.asr_sidecar import asr_sidecar_path
    au = case.get("audio_path")
    if au:
        p = asr_sidecar_path(au, f".asr_{asr_track}.txt", mirror_dir)
        try:
            t = Path(p).read_text(encoding="utf-8").strip()
            if t:
                return t
        except OSError:
            pass
    tp = case.get("text_path")
    if tp:
        try:
            return Path(tp).read_text(encoding="utf-8").strip()
        except OSError:
            pass
    return case.get("transcript") or ""


def _emit(case, transcript, label, slice_name, extra=None):
    cm = case.get("meta") or {}
    scen = case.get("scenario") or cm.get("scenario_type") or cm.get("scenario", "")   # preserve real FSS scenario type
    r = {"call_id": case.get("case_id", case.get("call_id")), "label": int(label),
         "source": case.get("source", "?"), "split": "test", "transcript": transcript,
         "audio_uri": _audio(case), "slice": slice_name, "scenario": scen,
         "meta": {"n_words": len(transcript.split()), "density": round(_density(transcript), 2),
                  "orig_density": round(_density(case.get("_orig_transcript", transcript)), 2),
                  "scenario": scen, **(extra or {})}}
    return r


def compose(cases, per_slice, seed, asr_track, mirror_dir, len_lo=150, len_hi=360, fair_audio=False):
    rng = random.Random(seed)
    # attach transcript and density
    pool = []
    for c in cases:
        t = _read_transcript(c, asr_track, mirror_dir)
        nw = len(t.split())
        if nw < len_lo or nw > len_hi:
            continue
        c = {**c, "_t": t, "_d": _density(t), "_nw": nw}
        pool.append(c)
    harm = [c for c in pool if _is_harm(c)]
    benign = [c for c in pool if not _is_harm(c)]
    if not harm and not benign:
        raise SystemExit("Insufficient pool: harm=0 benign=0 (after transcript/length filter) — check bundle")
    print(f"[hard] Real pool (passed transcript/length): harm={len(harm)} benign={len(benign)}", flush=True)
    ben_p90 = _pct([c["_d"] for c in benign], 90) if benign else 0.0
    harm_hi = sorted(harm, key=lambda c: -c["_d"])
    harm_lo = [c for c in sorted(harm, key=lambda c: c["_d"]) if c["_d"] <= max(ben_p90, 2.0)]
    ben_hi = sorted(benign, key=lambda c: -c["_d"])       # high-density benign = hard-benign
    ben_lo = sorted(benign, key=lambda c: c["_d"])        # low-density everyday = easy-benign

    out, used = [], set()

    def take(lst, k, label, name, degrade=False, reuse=False, suffix=""):
        n = 0
        for c in lst:
            if n >= k:
                break
            cid = c.get("case_id", c.get("call_id"))
            if not reuse and cid in used:
                continue
            if not reuse:
                used.add(cid)
            n += 1
            t = c["_t"]; extra = None
            c2 = {**c, "case_id": f"{cid}{suffix}", "_orig_transcript": t}
            if degrade:
                t = _asr_degrade(t, rng); extra = {"degraded": True}
            out.append(_emit(c2, t, label, name, extra))
        return n

    half = per_slice // 2
    n_eh = take(harm_hi, per_slice, 1, "easy-harm")                # high-density phishing (up to whole real pool)
    take(harm_lo, half, 1, "hard-harm-natural")                    # naturally low-density phishing (if any)
    take(harm_hi, half, 1, "hard-harm-asr", degrade=True, reuse=True, suffix="_asr")  # ASR-degraded (duplication allowed)
    take(ben_lo, per_slice, 0, "easy-benign")                      # low-density everyday
    take(ben_hi, per_slice, 0, "hard-benign")                      # high-density benign (legacy FP)
    if fair_audio:                                                  # source equalization (docs/BENCHMARK.md): real harm slices also get FSS -> benign cold-pool swap
        cold_carriers, _ = _split_cold_pool([_audio(c) for c in benign if _audio(c)])
        hi = 0
        for r in out:
            if int(r["label"]) == 1 and cold_carriers:             # harm carriers become benign cold recordings (phishing signal = text + cold prosody, not source)
                r["audio_uri"] = cold_carriers[hi % len(cold_carriers)]; hi += 1
            r.setdefault("meta", {})["audio_fair"] = True
        if cold_carriers:
            print(f"[hard] fair-audio: replaced carriers of {hi} real harm-slice cases with the benign cold pool (removes residual FSS leakage)", flush=True)
    rng.shuffle(out)
    for i, r in enumerate(out):
        r["order_idx"] = i
    return out


def _load_jsonl(path):
    """Defensive loader for large jsonl — skips broken lines (partial sync/corruption). Line-wise streaming."""
    rows, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"  ⚠️ {Path(path).name}: skipped {bad} unparseable lines ({len(rows)} valid)", flush=True)
    if not rows:
        raise SystemExit(f"Inventory has 0 valid lines — file likely corrupted: {path}")
    return rows


def _pct(xs, p):
    import numpy as np
    return float(np.percentile(xs, p)) if xs else 0.0


def _rank_cold(audio_uris):
    """Sort FSS audio by coldness (higher spectral_tilt = colder) — ensures prosody-transfer validity
    (only cold voices carry harm).

    docs/BENCHMARK.md: random pairing bound 58% of synth-hard-harm to non-cold audio, so XM never
    fired (mislabeling). Cold-selection corrects this.
    """
    from miltl.nibble.audio_decode import decode_to_pcm
    from miltl.nibble.prosody import prosody_stream
    scored = []
    for u in audio_uris:
        try:
            pcm, _ = decode_to_pcm(u, sr=16000)
            ps = prosody_stream(pcm, 16000, 8.0)
            tilt = float(np.mean([p.spectral_tilt for p in ps])) if ps else 0.0
        except Exception:                                  # noqa: BLE001
            tilt = -99.0                                   # decode failure = last priority
        scored.append((tilt, u))
    scored.sort(reverse=True)                              # highest tilt (coldest) first
    return [u for _, u in scored if _ > -99.0]


def _split_cold_pool(audio_uris):
    """Sort a same-source pool by coldness, then split into a cold half (harm carriers) and a
    warm half (benign carriers).

    Fair audio (docs/BENCHMARK.md): harm and benign carriers come from the **same corpus** and
    contrast only in prosody (cold/warm) -> forces audio-only models such as Wave-Seq to
    discriminate on actual prosody rather than the source corpus (removes source leakage).
    """
    ranked = _rank_cold(audio_uris)                        # coldest-first
    if len(ranked) < 2:
        return ranked, ranked
    half = max(1, len(ranked) // 2)
    return ranked[:half], ranked[half:]                    # (cold-carriers, warm-carriers)


def pair_synth(synth_cases, cases, seed, cold_select=False, cold_benign=False, fair_audio=False):
    """Prosody transfer (docs/BENCHMARK.md): pair synthetic text with real audio (target prosody).

    Default: cold (synthetic phishing) -> real FSS audio (coercive prosody);
             warm (synthetic benign) -> real benign audio.
    Synthetic text is kept; only audio_uri points to real audio. Legacy = text (decorrelated) -> collapse;
    MiLTL = decides via audio prosody.
    cold_select=True: sort FSS audio by coldness and assign coldest first (validity fix, docs/BENCHMARK.md).
    cold_benign=True (docs/BENCHMARK.md): assign cold real benign audio to synthetic benign =
    'cold benign call' hard negative.

    fair_audio=True (docs/BENCHMARK.md, source-leakage fix): draw harm and benign carriers
      **only from the same benign corpus** (cold half = harm, warm half = benign). Removes the
      source<->label collinearity of harm=FSS / benign=normal -> audio-only models must
      discriminate on actual (cold) prosody instead of a corpus shortcut.
      cold_select/cold_benign are ignored (fair takes precedence).
    """
    rng = random.Random(seed + 7)
    fss_au = [_audio(c) for c in cases if _is_harm(c) and _audio(c)]
    ben_au = [_audio(c) for c in cases if not _is_harm(c) and _audio(c)]
    if fair_audio:                                         # source equalization: all carriers from benign corpus, contrast only in prosody
        cold_pool, warm_pool = _split_cold_pool(ben_au)
        fss_au, ben_au = cold_pool, warm_pool
        print(f"[hard] fair-audio: harm and benign carriers both from benign corpus "
              f"(cold {len(cold_pool)} / warm {len(warm_pool)}) — source leakage removed", flush=True)
    elif cold_select:
        fss_au = _rank_cold(fss_au)                        # coldness order (not random)
        print(f"[hard] cold-select: sorted {len(fss_au)} FSS audio by coldness", flush=True)
    else:
        rng.shuffle(fss_au)
    if not fair_audio and cold_benign:
        ben_au = _rank_cold(ben_au)                        # coldest benign first = cold benign-call hard negative
        print(f"[hard] cold-benign: sorted {len(ben_au)} benign audio by coldness (cold normal-call transfer)", flush=True)
    elif not fair_audio:
        rng.shuffle(ben_au)
    out = []
    fi = bi = 0
    for sc in synth_cases:
        cold = sc.get("pair_prosody") == "cold"
        pool = fss_au if cold else ben_au
        if not pool:
            continue
        au = pool[(fi if cold else bi) % len(pool)]
        if cold:
            fi += 1
        else:
            bi += 1
        t = sc["transcript"]
        scen = sc.get("scenario") or (sc.get("meta") or {}).get("scenario", "")   # preserve scenario-type tag (post-hoc per-scenario analysis)
        out.append({"call_id": sc["case_id"], "label": int(sc["label"]), "source": "synth",
                    "split": "test", "transcript": t, "audio_uri": au, "slice": sc["slice"],
                    "scenario": scen,
                    "meta": {"n_words": len(t.split()), "density": round(_density(t), 2),
                             "orig_density": round(_density(t), 2), "synth": True, "scenario": scen,
                             "pair_prosody": sc.get("pair_prosody"), "audio_fair": bool(fair_audio)}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default="artifacts/manifest/kormmp_full.jsonl,artifacts/manifest/kormmp_hard_full.jsonl",
                    help="Real-pool bundles (comma-separated). Default=materialized inline-transcript bundles (no sidecar needed).")
    ap.add_argument("--per-slice", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--asr-track", default="000")
    ap.add_argument("--mirror-dir", default="artifacts/asr")
    ap.add_argument("--out", default="artifacts/manifest/kormmp_hardX_full.jsonl")
    ap.add_argument("--synth", default="", help="synth_edgecases.jsonl — add prosody-transfer pairing")
    ap.add_argument("--total", type=int, default=0, help="Total calls per round (per-seed random pick for variance). 0=all")
    ap.add_argument("--harm-ratio", type=float, default=0.4)
    ap.add_argument("--cold-select", action="store_true",
                    help="Pair synthetic phishing with cold FSS audio (prosody-transfer validity fix, docs/BENCHMARK.md)")
    ap.add_argument("--cold-benign", action="store_true",
                    help="Pair synthetic normal with cold real benign audio = cold normal-call hard negative (fairness, docs/BENCHMARK.md)")
    ap.add_argument("--transplant", action="store_true",
                    help="Prosody-transplant ablation (docs/BENCHMARK.md): replace harm carriers with the benign cold pool = full source-removal control condition")
    ap.add_argument("--fair-audio", action="store_true", help="Alias of --transplant (backward compatibility)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    args.transplant = args.transplant or args.fair_audio    # merge aliases
    if args.selftest:
        return _selftest()
    cases, seen = [], set()
    for p in args.inventory.split(","):
        p = p.strip()
        if not p or not Path(p).is_file():
            print(f"  ⚠️ bundle not found, skipping: {p}", flush=True)
            continue
        for c in _load_jsonl(p):
            cid = c.get("case_id", c.get("call_id"))
            if cid in seen:                              # remove cross-bundle duplicates (shared FSS)
                continue
            seen.add(cid); cases.append(c)
    print(f"[hard] Loaded {len(cases)} real-pool cases (deduplicated)", flush=True)
    out = compose(cases, args.per_slice, args.seed, args.asr_track, args.mirror_dir,
                  fair_audio=args.transplant)
    if args.synth and Path(args.synth).is_file():
        synth = _load_jsonl(args.synth)
        paired = pair_synth(synth, cases, args.seed, cold_select=args.cold_select,
                            cold_benign=args.cold_benign, fair_audio=args.transplant)
        out += paired
        for i, r in enumerate(out):
            r["order_idx"] = i
        print(f"[hard] Added {len(paired)} synthetic prosody-transfer cases", flush=True)
    if args.total and len(out) > args.total:               # per-seed random pick (variance management) + label balance
        rng2 = random.Random(args.seed + 1)
        harm = [r for r in out if r["label"] == 1]; ben = [r for r in out if r["label"] == 0]
        nh = min(len(harm), int(round(args.total * args.harm_ratio))); nb = min(len(ben), args.total - nh)
        rng2.shuffle(harm); rng2.shuffle(ben)
        out = harm[:nh] + ben[:nb]; rng2.shuffle(out)
        for i, r in enumerate(out):
            r["order_idx"] = i
        print(f"[hard] Random pick to {len(out)} total calls (harm={nh} benign={nb}, seed={args.seed})", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
    from collections import Counter
    cnt = Counter((r["slice"], r["label"]) for r in out)
    print(f"[hard] {len(out)} cases → {args.out}", flush=True)
    for k in sorted(cnt):
        ds = [r["meta"]["density"] for r in out if r["slice"] == k[0]]
        import numpy as np
        print(f"  {k[0]:<18} label={k[1]} n={cnt[k]:<3} lexical-density med={np.median(ds):.2f}", flush=True)
    return 0


def _selftest():
    # Synthetic inventory: high-density phishing / low-density phishing / low-density everyday / high-density benign
    scam = "계좌 이체 안전계좌 인증 비밀번호 대출 수수료"
    filler = "여보세요 네 그러니까 있잖아요 음 어 그래서 저기 그게 " * 30
    cases = []
    for i in range(20):
        cases.append({"case_id": f"h{i}", "class": "harm", "modality": "dual",
                      "transcript": f"{filler} {scam} {scam}"})              # high-density phishing
    for i in range(10):
        cases.append({"case_id": f"hl{i}", "class": "harm", "modality": "dual",
                      "transcript": f"{filler} 계좌 확인"})                    # low-density phishing
    for i in range(30):
        cases.append({"case_id": f"b{i}", "class": "benign", "modality": "dual",
                      "transcript": filler})                                  # low-density everyday
    for i in range(20):
        cases.append({"case_id": f"bh{i}", "class": "benign", "modality": "dual",
                      "transcript": f"{filler} {scam}"})                      # high-density benign (trap)
    out = compose(cases, per_slice=10, seed=1, asr_track="000", mirror_dir="")
    from collections import Counter
    import numpy as np
    cnt = Counter(r["slice"] for r in out)
    print(f"[selftest] slices: {dict(cnt)}")
    # does ASR degradation lower density?
    asr = [r for r in out if r["slice"] == "hard-harm-asr"]
    if asr:
        d0 = np.mean([r["meta"]["orig_density"] for r in asr]); d1 = np.mean([r["meta"]["density"] for r in asr])
        print(f"[selftest] ASR degradation: density {d0:.2f} → {d1:.2f} {'✅decreased' if d1 < d0 else '⚠️'}")
        assert d1 < d0
    hb = [r["meta"]["density"] for r in out if r["slice"] == "hard-benign"]
    eb = [r["meta"]["density"] for r in out if r["slice"] == "easy-benign"]
    print(f"[selftest] hard-benign density {np.mean(hb):.2f} > easy-benign {np.mean(eb):.2f} "
          f"{'✅trap' if np.mean(hb) > np.mean(eb) else '⚠️'}")
    assert np.mean(hb) > np.mean(eb)
    print("[selftest] Hard KorMMP composer works (lexicon-decorrelated slices + ASR degradation). Run=DGX (transcript sidecar).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
