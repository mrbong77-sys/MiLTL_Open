"""Acoustic neutrosophic head (learned) — architecture shared by training and inference (docs/BASELINES.md). Fills the NeutroWaveHead seam.

Design (docs/BASELINES.md): prosody feature sequence → distilled head → (T,I,F,E). AVD grounding —
  E=Arousal · F=high D + cold V (pressure) · I=high D + warm V (rapport) · T=balanced D + stable.
A rule that only looks at raw intensity (=Arousal) misses F (inverse direction, see design notes)
→ a learned head that looks at **D, V, and trajectory**.

Isomorphic to ko_engine: a shared architecture (build/train/save/load/score) blocks train/infer
mismatch. Pure numpy features → torch MLP. Trajectory features (neighbor diff, cumulative)
optionally inject sequence context.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

# Produced by prosody.py (AVD grounding, docs/BASELINES.md) — shared contract for synthetic/real. Fixed order (train/infer consistency).
FEATURES: List[str] = [
    "energy_mean", "energy_slope", "f0_mean", "f0_slope", "f0_std", "f0_range",  # A/D
    "rate_proxy", "jitter", "shimmer",                                            # A·F/stress
    "hnr_mean", "spectral_centroid", "spectral_tilt",                             # V
    "voiced_ratio", "pause_ratio", "pause_rate", "mean_pause_s",                  # I/hesitation
]
_CH = ("T", "I", "F", "E")


def feat_vec(p: dict) -> np.ndarray:
    return np.array([float(p.get(k, 0.0) or 0.0) for k in FEATURES], np.float32)


def with_trajectory(X: np.ndarray) -> np.ndarray:
    """Segment features (n,d) → +trajectory (previous diff, cumulative mean) = (n, 3d). Pressure lives in the trajectory (docs/BASELINES.md)."""
    d = X.shape[1]
    dif = np.vstack([np.zeros((1, d), np.float32), np.diff(X, axis=0)])
    cum = np.cumsum(X, axis=0) / np.arange(1, len(X) + 1)[:, None]
    return np.hstack([X, dif, cum]).astype(np.float32)


def build_head(in_dim: int, hidden: int = 64):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(hidden, 32), nn.ReLU(),
        nn.Linear(32, 4), nn.Sigmoid(),          # T,I,F,E ∈[0,1] (independent)
    )


def train_head(X: np.ndarray, Y: np.ndarray, epochs: int = 400, lr: float = 1e-3,
               wd: float = 1e-4, seed: int = 0):
    """X(n,d) features → Y(n,4) T/I/F/E targets (distill). MSE. → (module, in_dim)."""
    import torch
    torch.manual_seed(seed)
    d = X.shape[1]; net = build_head(d)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.tensor(X.astype(np.float32)); Yt = torch.tensor(Y.astype(np.float32))
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(Xt), Yt)
        loss.backward(); opt.step()
    net.eval()
    return net, d


def save_head(path, net, in_dim: int, trajectory: bool):
    import torch
    torch.save({"state_dict": net.state_dict(), "in_dim": int(in_dim),
                "features": FEATURES, "trajectory": bool(trajectory)}, path)


class WaveHeadScorer:
    """For injecting as NeutroWaveHead.score_fn(prosody dict)->(T,I,F,E). Loads a trained head."""

    def __init__(self, head_pt: str):
        import torch
        ck = torch.load(head_pt, map_location="cpu")
        self._traj = ck.get("trajectory", False)
        self._net = build_head(ck["in_dim"]); self._net.load_state_dict(ck["state_dict"]); self._net.eval()
        self._torch = torch
        self._hist: List[np.ndarray] = []

    def __call__(self, prosody: dict):
        x = feat_vec(prosody)
        if self._traj:                                   # streaming trajectory (previous diff, cumulative mean)
            self._hist.append(x)
            H = np.array(self._hist)
            x = with_trajectory(H)[-1]
        with self._torch.no_grad():
            o = self._net(self._torch.tensor(x[None, :]))[0]
        return (float(o[0]), float(o[1]), float(o[2]), float(o[3]))
