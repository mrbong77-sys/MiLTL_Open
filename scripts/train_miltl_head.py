#!/usr/bin/env python3
"""MiLTL trained head (docs/BENCHMARK.md) — channels [L,5]=(T,I,F,E,XM) → harm, TSMixer.

Breaks the interpretable-formula ceiling (decorrelated 0.69): nonlinear training absorbs the
per-call XM variance.
**Freeze compliance**: training = KorCCViD-train (real) + **separate synthetic hard set
(seed != eval, not KorMMP)**. Evaluation = frozen KorMMP.
Synthetic hard-harm = scam-lexicon-avoiding text + KorCCViD FSS audio (prosody transfer)
-> the head learns XM -> harm.

  # 1) separate synthetic text (seed 99, distinct from eval seed 42):
  python scripts/synth_edgecases.py --n-per 60 --seed 99 --out artifacts/manifest/synth_train.jsonl
  # 2) train the head (reuse KorCCViD cache + synthetic prosody transfer):
  python scripts/train_miltl_head.py --manifest artifacts/manifest/native_trainset.jsonl \
     --synth artifacts/manifest/synth_train.jsonl --ckpt artifacts/models/channel_extractors.pt \
     --cache artifacts/cache/chan_train300.npz --out artifacts/models/miltl_head.pt --device cuda
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.train_channel_extractors import _load_manifest, _stratified, _featurize_rows, _decode_pcm, _auroc
from miltl.native.channel_calib import Calib, fit_calib, channels5


def _synth_pair(synth_rows, real_items, seed):
    """Synthetic text + real KorCCViD audio (prosody transfer). cold->FSS, warm->benign. Returns NibbleChannelInput."""
    from miltl.native.nibble_features import featurize_channels
    rng = random.Random(seed)
    fss = [r for r, n in real_items if int(r.get("harm", 0)) == 1]
    ben = [r for r, n in real_items if int(r.get("harm", 0)) == 0]
    rng.shuffle(fss); rng.shuffle(ben)
    out = []
    fi = bi = 0
    for s in synth_rows:
        cold = s.get("pair_prosody") == "cold"
        pool = fss if cold else ben
        if not pool:
            continue
        src = pool[(fi if cold else bi) % len(pool)]
        if cold:
            fi += 1
        else:
            bi += 1
        pcm = _decode_pcm(src)
        if pcm is None:
            continue
        nci = featurize_channels(pcm, s["transcript"].split())
        out.append(({"harm": int(s["label"])}, nci))
    return out


def _build_xy(items, cal, device):
    import torch
    X = np.stack([channels5(n, cal) for _, n in items]).astype(np.float32)   # [N,L,5]
    y = np.array([int(r.get("harm", 0)) for r, _ in items], np.float32)
    mask = np.stack([n.mask for _, n in items]).astype(np.float32)
    return torch.tensor(X, device=device), torch.tensor(y, device=device), torch.tensor(mask, device=device), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--synth", default="")
    ap.add_argument("--ckpt", default="artifacts/models/channel_extractors.pt")
    ap.add_argument("--cache", default="")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out", default="artifacts/models/miltl_head.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    import torch
    from miltl.native.seq_head import TSMixerTiny

    # 1) featurize KorCCViD-train (real), reusing the cache
    rows = [r for r in _load_manifest(args.manifest) if r.get("split") == "train"]
    if args.limit:
        rows = _stratified(rows, args.limit)
    real_items = _featurize_rows(rows, None, None, cache=args.cache or None)
    print(f"[head] KorCCViD-train {len(real_items)} calls", flush=True)

    # 2) calib: checkpoint takes priority (matches the adapter), else fit on benign
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False) if Path(args.ckpt).is_file() else {}
    cal = Calib.from_dict(ck["calib"]) if "calib" in ck else fit_calib([n for r, n in real_items if int(r.get("harm", 0)) == 0])

    # 3) add the separate synthetic hard set (prosody transfer)
    items = list(real_items)
    if args.synth and Path(args.synth).is_file():
        synth_rows = _load_manifest(args.synth)
        syn = _synth_pair(synth_rows, real_items, args.seed)
        items += syn
        print(f"[head] added {len(syn)} synthetic hard-set calls (prosody transfer) → total {len(items)}", flush=True)

    # 4) [L,5] + labels -> train/val split
    Xa, ya, ma, y_np = _build_xy(items, cal, args.device)
    N = Xa.shape[0]
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    perm = torch.randperm(N, generator=g).to(args.device)
    nval = int(N * args.val_frac)
    vi, ti = perm[:nval], perm[nval:]

    m = TSMixerTiny(in_ch=5).to(args.device)
    pos = float(ya[ti].sum()); neg = len(ti) - pos
    pw = torch.tensor([neg / max(pos, 1)], device=args.device)
    opt = torch.optim.Adam(m.parameters(), lr=args.lr, weight_decay=1e-4)
    import torch.nn.functional as Fn
    best_au, best_state = -1.0, None
    bs = min(64, len(ti))
    for ep in range(args.epochs):
        m.train(); pp = ti[torch.randperm(len(ti), device=args.device)]
        for s in range(0, len(ti), bs):
            idx = pp[s:s + bs]
            loss = Fn.binary_cross_entropy_with_logits(m(Xa[idx], ma[idx]), ya[idx], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            va = torch.sigmoid(m(Xa[vi], ma[vi])).cpu().numpy()
        au = _auroc(va, y_np[vi.cpu().numpy()])
        if au > best_au:
            best_au = au; best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        if ep == 0 or (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep+1}/{args.epochs} val_AUROC={au:.3f} (best {best_au:.3f})", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "kind": "tsmixer", "in_ch": 5, "calib": cal.to_dict(),
                "val_auroc": best_au}, args.out)
    print(f"[head] saved {args.out} · best val_AUROC={best_au:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
