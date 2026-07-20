"""Sequence adaptor — trains a multi-scale 1D CNN on **continuous/cumulative/trajectory**
representations of the 15-segment stream (see design notes).

Core motivation: binary nibbles have low resolution (segment distributions overlap ~90%) and
kill benign/harm patterns. So the segment representation itself becomes a **design variable** —
binary / continuous / softmax (8-second competitive distribution) / zscore (distribution
standardization) / quantile — and a CNN adaptor lets the data empirically pick which
representation separates best. Beyond "simple binary T/I/F/E counts", the input carries the
continuous array + cumulative intensity (running mean) + trajectory (delta).

Pure numpy. PEINN untouched — the stored continuous tife is consumed with only its
representation changed. A trainable adaptor (not threshold recalibration).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .encoder import NibbleThresholds

_CH = ("T", "I", "F", "E")
ENCODINGS = ("binary", "continuous", "softmax", "zscore", "quantile")


# ============================================================ representation context
@dataclass
class FeatureContext:
    """Training-set statistics needed by the representations (thresholds, per-channel mean/std, quantile table). Computed at fit time, referenced by the representations."""
    thr: NibbleThresholds
    mean: dict = field(default_factory=dict)      # ch -> mean (zscore)
    std: dict = field(default_factory=dict)       # ch -> standard deviation (zscore)
    sorted_vals: dict = field(default_factory=dict)  # ch -> sorted values (quantile rank)
    softmax_temp: float = 0.5

    @classmethod
    def fit(cls, calls, thr: NibbleThresholds, modality: str = "text", softmax_temp: float = 0.5):
        attr = "wave_tife" if modality == "wave" else "text_tife"
        vals = {c: [] for c in _CH}
        for cs in calls:
            for s in cs.segments:
                tf = getattr(s, attr)
                if tf is not None:
                    for c in _CH:
                        vals[c].append(getattr(tf, c))
        mean = {c: (sum(v) / len(v) if v else 0.0) for c, v in vals.items()}
        std = {c: (math.sqrt(sum((x - mean[c]) ** 2 for x in v) / len(v)) if v else 1.0) or 1.0
               for c, v in vals.items()}
        return cls(thr=thr, mean=mean, std=std,
                   sorted_vals={c: sorted(v) for c, v in vals.items()}, softmax_temp=softmax_temp)


def _quantile(sorted_vals: Sequence[float], x: float) -> float:
    import bisect
    if not sorted_vals:
        return 0.5
    return bisect.bisect_right(sorted_vals, x) / len(sorted_vals)


def encode_segment(tife, encoding: str, ctx: FeatureContext) -> List[float]:
    """One segment's [T,I,F,E] -> vector (4-d) in the selected representation. Missing segments are zeroed by the caller."""
    v = [getattr(tife, c) for c in _CH]
    if encoding == "binary":
        return [1.0 if v[i] >= getattr(ctx.thr, f"tau_{_CH[i]}") else 0.0 for i in range(4)]
    if encoding == "continuous":
        return v
    if encoding == "softmax":
        z = [vi / ctx.softmax_temp for vi in v]
        m = max(z)
        e = [math.exp(zi - m) for zi in z]
        s = sum(e) or 1.0
        return [ei / s for ei in e]
    if encoding == "zscore":
        return [(v[i] - ctx.mean[_CH[i]]) / ctx.std[_CH[i]] for i in range(4)]
    if encoding == "quantile":
        return [_quantile(ctx.sorted_vals[_CH[i]], v[i]) for i in range(4)]
    raise ValueError(f"unknown encoding: {encoding}")


def sequence_matrix(cs, ctx: FeatureContext, encoding: str = "continuous",
                    groups: Tuple[str, ...] = ("seg", "cum"), L: int = 15,
                    modality: str = "text") -> np.ndarray:
    """Call -> (L, C) representation matrix. Most recent L segments, zero-padded at the front.

    groups: 'seg' (representation), 'cum' (running mean of continuous values = cumulative intensity),
    'delta' (trajectory diff). Each group is 4-d (per modality). A 1-d mask is always included.
    Cumulative/trajectory are computed on continuous values (representation-independent physical quantities)."""
    attr = "wave_tife" if modality == "wave" else "text_tife"
    segs = [getattr(s, attr) for s in cs.segments]
    segs = [tf for tf in segs][-L:]
    n = len(segs)
    off = L - n
    # continuous values (for cumulative/trajectory computation)
    cont = [[getattr(tf, c) for c in _CH] if tf is not None else None for tf in segs]
    per = 4 * len(groups) + 1
    M = np.zeros((L, per), dtype=np.float32)
    run = [0.0, 0.0, 0.0, 0.0]
    cnt = 0
    prev = None
    for i, tf in enumerate(segs):
        r = off + i
        col = 0
        if tf is None:
            prev = None
            continue
        if "seg" in groups:
            M[r, col:col + 4] = encode_segment(tf, encoding, ctx); col += 4
        cv = cont[i]
        if "cum" in groups:
            cnt += 1
            run = [run[j] + cv[j] for j in range(4)]
            M[r, col:col + 4] = [run[j] / cnt for j in range(4)]; col += 4   # cumulative intensity (running mean)
        if "delta" in groups:
            M[r, col:col + 4] = [0.0] * 4 if prev is None else [cv[j] - prev[j] for j in range(4)]
            col += 4
        M[r, per - 1] = 1.0     # mask
        prev = cv
    return M


# ============================================================ multi-scale 1D CNN
def _sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


@dataclass
class MultiScaleCNNAdaptor:
    """CANONICAL Gate-1 (see design notes and canonical.py) — the evaluated System-1 call scorer.

    Multi-kernel 1D-CNN + (max and avg) pooling -> logistic. Captures local (short bursts) through
    long-range (sustained pressure) patterns simultaneously. Used by MiLTLDetector (text/wave/dual).
    Winner of the 4-candidate model selection (AUROC 0.925). The other variants are comparison
    baselines (RNN/Conv/GRU), operational (Rule/MM), or experimental (forecast) — role
    classification rather than deletion (canonical.py).

    Input: (L, C) representation matrix (sequence_matrix). Per kernel: conv -> ReLU ->
    [maxpool|avgpool], then concat everything. Pure numpy forward/backward + SGD.
    score_call(cs) decides a call."""
    kernels: Tuple[int, ...] = (2, 3, 5)
    K: int = 12
    L: int = 15
    C: int = 9
    lr: float = 0.03
    l2: float = 1e-4
    epochs: int = 60
    seed: int = 0
    encoding: str = "continuous"
    groups: Tuple[str, ...] = ("seg", "cum")
    modality: str = "text"
    class_balance: bool = True         # weight positives (neg/pos) on imbalanced training sets. No effect when balanced.
    class_balance_cap: float = 4.0     # cap on positive weighting — prevents over-weighting (-> harm bias = FP explosion). min with neg/pos.
    banks: Optional[list] = None       # [(ks, W(ks*C,K), b(K)), ...]
    wo: Optional[np.ndarray] = None    # (len(kernels)*2*K,)
    bo: float = 0.0
    ctx: Optional[FeatureContext] = None

    def _init(self, rng):
        self.banks = []
        for ks in self.kernels:
            W = (rng.standard_normal((ks * self.C, self.K)) * 0.1).astype(np.float32)
            b = np.zeros(self.K, dtype=np.float32)
            self.banks.append([ks, W, b])
        self.wo = (rng.standard_normal(len(self.kernels) * 2 * self.K) * 0.1).astype(np.float32)
        self.bo = 0.0

    def _forward(self, M):
        feats, cache = [], []
        for ks, W, b in self.banks:
            P = self.L - ks + 1
            X = np.stack([M[p:p + ks].reshape(-1) for p in range(P)])   # (P, ks*C)
            z = X @ W + b                                               # (P, K)
            a = np.maximum(z, 0.0)
            mxarg = a.argmax(axis=0)
            mx = a[mxarg, np.arange(self.K)]                            # (K,)
            av = a.mean(axis=0)                                         # (K,)
            feats.append(mx); feats.append(av)
            cache.append((X, z, a, mxarg, P))
        f = np.concatenate(feats)                                       # (2K*nk,)
        logit = float(f @ self.wo + self.bo)
        return _sigmoid(logit), (f, cache)

    def _backward(self, M, cache_pack, d):
        f, cache = cache_pack
        gwo = d * f + self.l2 * self.wo
        gbo = d
        df = d * self.wo                                               # (2K*nk,)
        for bi, (ks, W, b) in enumerate(self.banks):
            X, z, a, mxarg, P = cache[bi]
            dmx = df[bi * 2 * self.K: bi * 2 * self.K + self.K]
            dav = df[bi * 2 * self.K + self.K: (bi + 1) * 2 * self.K]
            da = np.zeros_like(a)
            da[mxarg, np.arange(self.K)] += dmx                        # max-pool routing
            da += dav / P                                              # avg-pool distribution
            dz = da * (z > 0)
            gW = X.T @ dz + self.l2 * W
            gb = dz.sum(axis=0)
            W -= self.lr * gW
            b -= self.lr * gb
        self.wo -= self.lr * gwo
        self.bo -= self.lr * gbo

    def fit(self, calls, labels, ctx: Optional[FeatureContext] = None):
        self.ctx = ctx or FeatureContext.fit(
            calls, NibbleThresholds(), self.modality)   # representation statistics
        Ms = [sequence_matrix(cs, self.ctx, self.encoding, self.groups, self.L, self.modality)
              for cs in calls]
        return self.fit_matrices(Ms, labels)

    def fit_matrices(self, Ms, labels):
        """Train directly on prebuilt (L,C) matrices (bypasses sequence_matrix — reused for e.g. wave prosody sequences)."""
        rng = np.random.default_rng(self.seed)
        self.C = Ms[0].shape[1] if Ms else self.C
        self._init(rng)
        y = np.asarray(labels, np.float32)
        # Class balancing (for imbalanced training sets): weight positive gradients by the neg/pos ratio (prevents harm under-fit, capped).
        npos = float(y.sum()); nneg = float(len(y) - npos)
        w_pos = min(nneg / npos, self.class_balance_cap) if (self.class_balance and npos > 0) else 1.0
        idx = np.arange(len(Ms))
        for _ in range(self.epochs):
            rng.shuffle(idx)
            for i in idx:
                p, pack = self._forward(Ms[i])
                w = w_pos if y[i] == 1.0 else 1.0
                self._backward(Ms[i], pack, (p - y[i]) * w)
        return self

    def score_matrix(self, M) -> float:
        """Score a prebuilt (L,C) matrix directly (e.g. wave prosody)."""
        return float(self._forward(M)[0])

    def predict_proba_call(self, cs) -> float:
        M = sequence_matrix(cs, self.ctx, self.encoding, self.groups, self.L, self.modality)
        return float(self._forward(M)[0])

    def score_call(self, cs) -> float:
        return self.predict_proba_call(cs)

    # ---- save/load (JSON) ----
    def to_dict(self) -> dict:
        return {"arch": "multiscale_cnn", "kernels": list(self.kernels), "K": self.K, "L": self.L,
                "C": self.C, "encoding": self.encoding, "groups": list(self.groups),
                "modality": self.modality,
                "banks": [[ks, W.tolist(), b.tolist()] for ks, W, b in self.banks],
                "wo": self.wo.tolist(), "bo": self.bo,
                # ctx = text encoding statistics (for sequence_matrix). wave (fit_matrices) has no ctx -> null.
                "ctx": None if self.ctx is None else {
                        "tau": {c: getattr(self.ctx.thr, f"tau_{c}") for c in _CH},
                        "mean": self.ctx.mean, "std": self.ctx.std,
                        "softmax_temp": self.ctx.softmax_temp,
                        "sorted_vals": {c: self.ctx.sorted_vals[c] for c in _CH}}}

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "MultiScaleCNNAdaptor":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, d: dict) -> "MultiScaleCNNAdaptor":
        m = cls(kernels=tuple(d["kernels"]), K=d["K"], L=d["L"], C=d["C"],
                encoding=d["encoding"], groups=tuple(d["groups"]), modality=d["modality"])
        m.banks = [[ks, np.asarray(W, np.float32), np.asarray(b, np.float32)]
                   for ks, W, b in d["banks"]]
        m.wo = np.asarray(d["wo"], np.float32); m.bo = float(d["bo"])
        cc = d.get("ctx")
        if cc:                                              # text: restore encoding statistics. wave (ctx null): score_matrix only.
            thr = NibbleThresholds(tau_T=cc["tau"]["T"], tau_I=cc["tau"]["I"],
                                   tau_F=cc["tau"]["F"], tau_E=cc["tau"]["E"], use_conflict=False)
            m.ctx = FeatureContext(thr=thr, mean=cc["mean"], std=cc["std"],
                                   sorted_vals=cc["sorted_vals"], softmax_temp=cc["softmax_temp"])
        return m


def train_cnn_adaptor(calls, labels, encoding: str = "continuous",
                      groups: Tuple[str, ...] = ("seg", "cum"), thr: Optional[NibbleThresholds] = None,
                      modality: str = "text", **kw) -> MultiScaleCNNAdaptor:
    """List of calls -> trained CNN adaptor. If thr is given, it is used as the binary-representation threshold."""
    ctx = FeatureContext.fit(calls, thr or NibbleThresholds(), modality)
    m = MultiScaleCNNAdaptor(encoding=encoding, groups=groups, modality=modality, **kw)
    return m.fit(calls, labels, ctx)


# Architecture registry (spine is cnn-only) — for RNN comparison baselines see archive.rnn_adaptor.ADAPTORS_ALL.
ADAPTORS = {"cnn": MultiScaleCNNAdaptor}
