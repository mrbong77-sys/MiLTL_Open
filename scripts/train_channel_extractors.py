#!/usr/bin/env python3
"""Channel-extractor training + discriminability probe (docs/ARCHITECTURE.md, decision D-C) — train and validate the extractors first.

**Core gate (deterministic)**: before training the head, verify from data that each channel carries its own signal.
  1) --probe : measure channel discriminability at the teacher-target level (before extractor training).
               F(fss) >> F(benign)?  I early > late (fss)?  E ⊥ harm (≈0.5)?  ← is the signal there at all.
  2) train   : regress ChannelExtractors onto the teacher (T,I,F,E) targets (valid-nibble MSE).
  3) post    : re-measure discriminability at the extractor-output level (preserves/exceeds the teacher?). If it passes, proceed to the head (seq_head).

Input: build_native_trainset.py manifest (jsonl).
  {call_id, audio_uri|audio_uris, utterances, harm, corpus:"fss|callcenter|benign", split}

**DGX-gate**: requires real audio decoding (prosody). Locally use --selftest (synthetic). Run:
  python scripts/train_channel_extractors.py --manifest artifacts/manifest/native_trainset.jsonl \
     --probe --split train --limit 80              # Gate-1: target discriminability
  python scripts/train_channel_extractors.py --manifest ... --epochs 20 \
     --out artifacts/models/channel_extractors.pt  # training + Gate-3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from miltl.native.nibble_features import featurize_channels, PROS_DIM
from miltl.native.channel_calib import fit_calib, channels, Calib

CH = ("T", "I", "F", "E")
_TRAPZ = getattr(np, "trapezoid", getattr(np, "trapz", None))


def _auroc(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels, float)
    order = np.argsort(-scores)
    y = labels[order]
    P, N = y.sum(), len(y) - y.sum()
    if P == 0 or N == 0:
        return float("nan")
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    return float(_TRAPZ(tp / P, fp / N))


def _load_manifest(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _evenly(lst, k):
    if len(lst) <= k:
        return list(lst)
    step = len(lst) / k
    return [lst[int(i * step)] for i in range(k)]


def _stratified(rows, limit):
    """Harm/benign-balanced sample (evenly spaced). Prevents single-class manifest ordering bias."""
    harm = [r for r in rows if int(r.get("harm", 0)) == 1]
    ben = [r for r in rows if int(r.get("harm", 0)) == 0]
    k = limit // 2
    return _evenly(harm, k) + _evenly(ben, limit - k)


def _decode_pcm(row, sr=16000):
    from miltl.nibble.audio_decode import decode_to_pcm
    uris = row.get("audio_uris") or ([row["audio_uri"]] if row.get("audio_uri") else [])
    pcms = []
    for u in uris:
        try:
            x, _ = decode_to_pcm(u, sr=sr)
            pcms.append(x)
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️ decode failed {u}: {type(e).__name__}", flush=True)
    return np.concatenate(pcms).astype(np.float32) if pcms else None


def _featurize_rows(rows, text_enc, sa_clf, cache=None):
    """rows → [(row, NibbleChannelInput)]. Targets are computed later via channel_calib (benign fit)."""
    if cache and Path(cache).is_file():
        print(f"[chan] cache load {cache}", flush=True)
        d = np.load(cache, allow_pickle=True)
        return list(d["items"])
    out = []
    for i, r in enumerate(rows):
        pcm = _decode_pcm(r)
        if pcm is None and not r.get("utterances"):
            continue
        nci = featurize_channels(pcm, r.get("utterances", []), text_enc=text_enc, sa_clf=sa_clf,
                                 times=([tuple(t) for t in r["times"]] if r.get("times") else None))
        out.append((r, nci))
        if (i + 1) % 25 == 0:
            print(f"  featurize {i + 1}/{len(rows)}", flush=True)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        arr = np.empty(len(out), dtype=object)                # 1D object array (prevents tuple shape mis-inference)
        for i, v in enumerate(out):
            arr[i] = v
        np.savez(cache, items=arr)
        print(f"[chan] cache saved {cache} ({len(out)} calls)", flush=True)
    return out


def _call_channel_means(tife, mask, half=None):
    """Per-channel mean of tife[L,4] over valid nibbles. half='early'|'late' → first/second half only."""
    m = mask > 0.5
    idx = np.where(m)[0]
    if len(idx) == 0:
        return np.zeros(4, np.float32)
    if half == "early":
        idx = idx[: max(1, len(idx) // 2)]
    elif half == "late":
        idx = idx[len(idx) // 2:]
    return tife[idx].mean(0)


def _agg3(tife, mask):
    """Valid nibbles → (mean, early third, late third) [4 each] trajectory aggregates."""
    idx = np.where(mask > 0.5)[0]
    if len(idx) == 0:
        z = np.zeros(4, np.float32); return z, z, z
    v = tife[idx]
    k = max(1, len(idx) // 3)
    return v.mean(0), v[:k].mean(0), v[-k:].mean(0)


def _dynamics_score(tife, mask):
    """Composite trajectory-dynamics harm score (fixed direction, no learning) — docs/ARCHITECTURE.md.

    Threat = front-loaded transient state; benign = stationary. Components (all oriented fss↑):
      static F, I high, T low + transient (early−late) I, F high (front-loaded) + |slope| proxy (transient magnitude).
    """
    mean, early, late = _agg3(tife, mask)
    transient_I = early[1] - late[1]      # latency (I) dominant early (fss>0)
    transient_F = early[2] - late[2]      # coercion (F) dominant early
    return (mean[2] + mean[1] - mean[0]) + (transient_I + transient_F)


def _report_separation(items, channel_source, tag):
    """channel_source(nci)->tife[L,4]. Per-channel discriminability, trajectory, magnitude + composite dynamics AUROC. Returns dict."""
    harm, means, fss_early_I, fss_late_I, fss_early_F, fss_late_F = [], [], [], [], [], []
    ben_F, ben_I, fss_F_all, dyn, early_I_all, early_F_all = [], [], [], [], [], []
    for (r, nci) in items:
        tife = channel_source(nci)
        mu, ea, la = _agg3(tife, nci.mask)
        means.append(mu); harm.append(int(r["harm"]))
        dyn.append(_dynamics_score(tife, nci.mask))
        early_I_all.append(ea[1]); early_F_all.append(ea[2])
        if int(r["harm"]) == 0:
            ben_F.append(mu[2]); ben_I.append(mu[1])
        if r.get("corpus") == "fss":
            fss_early_I.append(ea[1]); fss_late_I.append(la[1])
            fss_early_F.append(ea[2]); fss_late_F.append(la[2]); fss_F_all.append(la[2])
    means = np.stack(means); harm = np.array(harm)
    benign = 1 - harm
    auroc = {
        "F→harm": _auroc(means[:, 2], harm),      # coercion: higher = harm
        "I→harm": _auroc(means[:, 1], harm),      # latency: harm direction
        "T→benign": _auroc(means[:, 0], benign),  # naturalness: benign direction
        "E→harm": _auroc(means[:, 3], harm),      # arousal: should be ≈0.5 (⊥harm)
    }
    print(f"\n=== [{tag}] channel discriminability (n={len(harm)}, harm={harm.sum()}, benign={benign.sum()}) ===", flush=True)
    for k, v in auroc.items():
        flag = ""
        if k == "E→harm":
            flag = "  ✅⊥" if abs(v - 0.5) < 0.15 else "  ⚠️arousal shortcut?"
        elif k in ("F→harm", "I→harm", "T→benign"):
            flag = "  ✅" if v >= 0.65 else ("  △" if v >= 0.55 else "  ✗weak")
        print(f"  AUROC {k:10s} = {v:.3f}{flag}", flush=True)
    # Dynamics-aware AUROC: adds early-window and transient info over the static mean (threat = front-loaded)
    au_dyn = _auroc(np.array(dyn), harm)
    au_eI = _auroc(np.array(early_I_all), harm); au_eF = _auroc(np.array(early_F_all), harm)
    print(f"  ── dynamics-aware ──", flush=True)
    print(f"  AUROC {'I early→harm':10s} = {au_eI:.3f}   (vs static I={auroc['I→harm']:.3f}, front-loading aware)", flush=True)
    print(f"  AUROC {'F early→harm':10s} = {au_eF:.3f}   (static F={auroc['F→harm']:.3f})", flush=True)
    print(f"  AUROC {'dynamics→harm':10s} = {au_dyn:.3f}   (F+I−T + transient, fixed direction, no learning)"
          f"  {'✅' if au_dyn >= max(auroc['F→harm'], auroc['I→harm']) else ''}", flush=True)
    auroc["dynamics"] = au_dyn; auroc["I_early"] = au_eI; auroc["F_early"] = au_eF
    # magnitude (zero-shot escape indicator): target benign F/I median ≈0.1, fss late F ≥0.5
    if ben_F:
        print(f"  magnitude: benign F_med={np.median(ben_F):.3f} I_med={np.median(ben_I):.3f}"
              f"  ·  fss F_late_med={np.median(fss_F_all):.3f}"
              f"  {'✅zero-shot escape' if np.median(fss_F_all) >= 0.4 else '⚠️still low activation'}", flush=True)
    traj = {}
    if fss_early_I:
        eI, lI = np.mean(fss_early_I), np.mean(fss_late_I)
        eF, lF = np.mean(fss_early_F), np.mean(fss_late_F)
        traj = {"I_early": eI, "I_late": lI, "F_early": eF, "F_late": lF}
        print(f"  FSS trajectory: I early={eI:.3f} {'>' if eI > lI else '≤'} late={lI:.3f}"
              f"  {'✅latency→early' if eI > lI else '⚠️'}", flush=True)
        print(f"           F early={eF:.3f} {'<' if eF < lF else '≥'} late={lF:.3f}"
              f"  {'✅coercion→late' if eF < lF else '⚠️'}", flush=True)
    return {"auroc": auroc, "traj": traj}


def _sa5(nci):
    """Lexical-signal extractor input [L,5] = speech_act[4] + warmth (cross-modal I term)."""
    w = nci.warmth if nci.warmth is not None else np.zeros(nci.mask.shape[0], np.float32)
    return np.concatenate([nci.speech_act, w[:, None]], axis=-1).astype(np.float32)


def _diag_components(items, cal):
    """Decompose each evidence component (prosody/lexicon derived) into fss vs benign — deterministically localize the signal.

    Why F/I invert: measure which component (D, threat, directive, warmth, XM, cold) actually separates.
    """
    from miltl.native.channel_calib import avd_from_z, evidence, _IX
    keys = ["A", "V", "D", "cold", "threat", "directive", "urgency", "subversion", "warmth", "XM", "zF", "zI", "zT"]
    vals = {k: [] for k in keys}
    harm = []
    for (r, nci) in items:
        m = nci.mask > 0.5
        if not m.any():
            continue
        z = cal.zfeat(nci.prosody)
        avd = avd_from_z(z)
        w = nci.warmth if nci.warmth is not None else np.zeros(len(z), np.float32)
        ev = evidence(avd, nci.speech_act, w, z[:, _IX["pause_ratio"]])
        from miltl.native.channel_calib import _coherence
        zT = ev["zT_local"] + 0.35 * _coherence(ev["A"][m])
        row = {"A": avd[m, 0].mean(), "V": avd[m, 1].mean(), "D": avd[m, 2].mean(),
               "cold": (1 - avd[m, 1]).mean(), "threat": nci.speech_act[m, 2].mean(),
               "directive": nci.speech_act[m, 0].mean(), "urgency": nci.speech_act[m, 1].mean(),
               "subversion": nci.speech_act[m, 3].mean(), "warmth": w[m].mean(),
               "XM": ev["XM"][m].mean(), "zF": ev["zF"][m].mean(), "zI": ev["zI"][m].mean(),
               "zT": zT[m].mean()}
        for k in keys:
            vals[k].append(row[k])
        harm.append(int(r["harm"]))
    harm = np.array(harm)
    print(f"\n=== [component breakdown] fss(harm={harm.sum()}) vs benign({len(harm)-harm.sum()}) — localize the signal ===", flush=True)
    print(f"  {'component':<12}{'benign_mean':>12}{'fss_mean':>12}{'AUROC→harm':>12}", flush=True)
    for k in keys:
        v = np.array(vals[k])
        bm, fm = v[harm == 0].mean(), v[harm == 1].mean()
        au = _auroc(v, harm)
        flag = "  ★separates" if (au >= 0.65 or au <= 0.35) else ""
        print(f"  {k:<12}{bm:>12.3f}{fm:>12.3f}{au:>12.3f}{flag}", flush=True)
    print("  (AUROC>0.65: fss-high signal · <0.35: benign-high = inverted signal · ≈0.5: no signal)", flush=True)


def _traj_dynamics(items, cal):
    """Channel-trajectory (cumulative/slope) dynamics — hypothesis: benign = T accumulates↑; threat = I persists/rises, T suppressed (docs/ARCHITECTURE.md).

    Linear-regression slope over per-call valid nibbles + (late half − early half) delta. fss vs benign means.
    """
    def slope(v):
        n = len(v)
        if n < 2:
            return 0.0
        x = np.arange(n) - (n - 1) / 2
        return float((x * (v - v.mean())).sum() / (x * x).sum())
    agg = {c: {"fss_slope": [], "ben_slope": [], "fss_delta": [], "ben_delta": []} for c in ("T", "I", "F")}
    for (r, nci) in items:
        m = nci.mask > 0.5
        idx = np.where(m)[0]
        if len(idx) < 3:
            continue
        tife = channels(nci, cal)[idx]
        half = len(idx) // 2
        for ci, c in zip((0, 1, 2), ("T", "I", "F")):
            v = tife[:, ci]
            sl = slope(v); dl = float(v[half:].mean() - v[:half].mean())
            k = "fss" if r.get("corpus") == "fss" else "ben"
            agg[c][f"{k}_slope"].append(sl); agg[c][f"{k}_delta"].append(dl)
    print("\n=== [trajectory dynamics] channel change over time (slope>0=rising, delta=late−early) — T-accumulation/I-latency hypothesis check ===", flush=True)
    print(f"  {'chan':<6}{'benign_slope':>14}{'fss_slope':>12}{'benign_Δ':>11}{'fss_Δ':>10}", flush=True)
    for c in ("T", "I", "F"):
        a = agg[c]
        print(f"  {c:<6}{np.mean(a['ben_slope']):>14.4f}{np.mean(a['fss_slope']):>12.4f}"
              f"{np.mean(a['ben_delta']):>11.3f}{np.mean(a['fss_delta']):>10.3f}", flush=True)
    print("  If the hypothesis holds: benign T_slope>0 (accumulation) · fss I_slope≥0 (latency persists/rises) · fss T_slope≤benign", flush=True)


def train(items, cal, epochs, lr, device, out):
    import torch
    from miltl.native.channels import ChannelExtractors, ChannelConfig
    pros = torch.tensor(np.stack([nci.prosody for _, nci in items]), device=device)
    text = torch.tensor(np.stack([nci.text for _, nci in items]), device=device)
    sa = torch.tensor(np.stack([_sa5(nci) for _, nci in items]), device=device)  # [.,.,5] +warmth
    mask = torch.tensor(np.stack([nci.mask for _, nci in items]), device=device)
    tgt = torch.tensor(np.stack([channels(nci, cal) for _, nci in items]), device=device)  # [N,L,4] calibrated targets
    m = ChannelExtractors(ChannelConfig()).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    N = pros.shape[0]; bs = min(32, N)
    print(f"[chan] extractor training N={N} epochs={epochs} params={m.num_params():,}", flush=True)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(N, device=device); tot = 0.0
        for s in range(0, N, bs):
            idx = perm[s:s + bs]
            tife, _ = m(pros[idx], text[idx], sa[idx])
            w = mask[idx].unsqueeze(-1)                       # valid nibbles only
            loss = ((tife - tgt[idx]) ** 2 * w).sum() / w.sum().clamp(min=1) / 4
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        if ep == 0 or (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"  epoch {ep + 1}/{epochs}  MSE={tot / N:.4f}", flush=True)
    m.eval()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": m.state_dict(), "cfg": vars(ChannelConfig()), "calib": cal.to_dict()}, out)
        print(f"[chan] saved {out} (+calibrator)", flush=True)
    return m


def _extractor_source(m, device):
    import torch
    def src(nci):
        with torch.no_grad():
            p = torch.tensor(nci.prosody[None], device=device)
            t = torch.tensor(nci.text[None], device=device)
            s = torch.tensor(_sa5(nci)[None], device=device)
            tife, _ = m(p, t, s)
        return tife[0].cpu().numpy()
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--probe", action="store_true", help="target discriminability only (no training)")
    ap.add_argument("--diag", action="store_true", help="evidence component breakdown (fss vs benign) — localize the signal")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default="")
    ap.add_argument("--cache", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--real-text", action="store_true", help="klue-roberta (slow); default is Mock")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()

    rows = _load_manifest(args.manifest)
    rows = [r for r in rows if r.get("split") == args.split]
    if args.limit:
        rows = _stratified(rows, args.limit)                 # harm/benign balance (avoids nan AUROC)
    nh = sum(int(r["harm"]) for r in rows)
    print(f"[chan] {args.split} {len(rows)} calls (limit={args.limit or '-'}) · harm={nh} benign={len(rows)-nh}", flush=True)
    if nh == 0 or nh == len(rows):
        print("  ⚠️ single class — AUROC=nan. Increase --limit or check the split.", flush=True)
    text_enc = None
    if args.real_text:
        from miltl.native.features import KlueRobertaEncoder
        text_enc = KlueRobertaEncoder(device=args.device)
    items = _featurize_rows(rows, text_enc, None, cache=args.cache or None)
    print(f"[chan] featurize done {len(items)} calls", flush=True)

    # Calibrator fit: benign only (frozen protocol) — defines activation thresholds and sensitivity (docs/ARCHITECTURE.md)
    benign_nci = [nci for (r, nci) in items if int(r["harm"]) == 0]
    if not benign_nci:
        print("  ⚠️ no benign — cannot calibrate. Check split/limit.", flush=True)
        return 1
    cal = fit_calib(benign_nci)
    print(f"[chan] Calib fit (benign {len(benign_nci)} calls): "
          f"thr F={tuple(round(x,2) for x in cal.thr['F'])} I={tuple(round(x,2) for x in cal.thr['I'])} "
          f"T={tuple(round(x,2) for x in cal.thr['T'])}", flush=True)

    if args.diag:
        _diag_components(items, cal)
        _traj_dynamics(items, cal)
        return 0

    # Gate-1: calibrated-channel (neutrosophic/affect) discriminability — is the signal there at all (before extractor training)
    _report_separation(items, lambda nci: channels(nci, cal), "calibrated channels (teacher)")
    if args.probe:
        print("\n[chan] --probe: reporting channel discriminability only. Train after confirming Gate-1 passes.", flush=True)
        return 0

    m = train(items, cal, args.epochs, args.lr, args.device, args.out)
    # Gate-3: extractor-output discriminability (preserves/exceeds the calibrated targets?)
    _report_separation(items, _extractor_source(m, args.device), "extractor output")
    return 0


def _selftest():
    import numpy as np
    from miltl.native.features import SR
    rng = np.random.default_rng(0)
    # Synthetic: 12 fss calls (threat/directive/high-energy) vs 12 benign calls (friendly/low-energy) — verifies calib/separation wiring
    def synth(freq, amp, sec=20):
        t = np.arange(int(SR * sec)) / SR
        return (amp * np.sin(2 * np.pi * freq * t) + 0.02 * rng.standard_normal(len(t))).astype(np.float32)
    fss_utts = ["고객님 안녕하세요 도와드릴게요", "네 확인 부탁드려요",
                "지금 즉시 검찰 수사 계좌 이체 하세요", "당장 안전계좌로 송금 안하면 구속입니다"]
    ben_utts = ["어제 뭐 했어 재밌었어", "그냥 집에서 쉬었지", "날씨 좋더라 산책하자", "응 그래 고마워"]
    rows = []
    for k in range(12):
        rows.append({"call_id": f"f{k}", "harm": 1, "corpus": "fss", "utterances": fss_utts,
                     "_pcm": synth(230 + k, 0.12)})
        rows.append({"call_id": f"b{k}", "harm": 0, "corpus": "benign", "utterances": ben_utts,
                     "_pcm": synth(175 + k, 0.05)})
    global _decode_pcm
    _decode_pcm = lambda row, sr=16000: row.get("_pcm")  # noqa: E731  inject synthetic pcm
    items = _featurize_rows(rows, None, None)
    assert items[0][1].prosody.shape[1] == PROS_DIM
    cal = fit_calib([nci for (r, nci) in items if int(r["harm"]) == 0])
    _report_separation(items, lambda nci: channels(nci, cal), "calibrated channels (synthetic)")
    m = train(items, cal, epochs=30, lr=5e-3, device="cpu", out="")
    _report_separation(items, _extractor_source(m, "cpu"), "extractor output (synthetic)")
    print("[selftest] calib + extractor training + discriminability probe wiring OK. Real discriminability = DGX (real audio).", flush=True)
    return 0


import scripts.train_channel_extractors as T  # noqa: E402  (self-reference for selftest monkeypatch)


if __name__ == "__main__":
    raise SystemExit(main())
