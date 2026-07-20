"""Trainable Gate-1 — pure-stdlib logistic regression (rule -> learned transition).

Learns P(phishing) from the fixed MMFeatures feature vector. TrainedGate1 exposes the same
.score(mm) interface as the rule-based MultimodalGate1, so it drops straight into
cascade/simulate. Weights are saved/loaded as JSON.

Zero dependencies (pure Python). On the DGX the same features could feed a larger model
(1D-CNN/GRU), but this logistic serves as the cold-start, ultra-light edge baseline and as
the check that learning beats the rules.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .mm_features import MMFeatures, FEATURE_NAMES, mm_feature_vector


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class LogisticRegression:
    """Standardization + L2-regularized batch gradient descent. Pure stdlib."""
    lr: float = 0.1
    l2: float = 1e-3
    epochs: int = 400
    mu: List[float] = field(default_factory=list)
    sigma: List[float] = field(default_factory=list)
    w: List[float] = field(default_factory=list)
    b: float = 0.0

    def _standardize_fit(self, X: Sequence[Sequence[float]]):
        n, d = len(X), len(X[0])
        self.mu = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
        var = [sum((X[i][j] - self.mu[j]) ** 2 for i in range(n)) / max(1, n) for j in range(d)]
        self.sigma = [math.sqrt(v) or 1.0 for v in var]

    def _z(self, x: Sequence[float]) -> List[float]:
        return [(x[j] - self.mu[j]) / self.sigma[j] for j in range(len(x))]

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> "LogisticRegression":
        n, d = len(X), len(X[0])
        self._standardize_fit(X)
        Z = [self._z(x) for x in X]
        self.w = [0.0] * d
        self.b = 0.0
        for _ in range(self.epochs):
            gw = [0.0] * d
            gb = 0.0
            for i in range(n):
                p = _sigmoid(sum(self.w[j] * Z[i][j] for j in range(d)) + self.b)
                err = p - y[i]
                for j in range(d):
                    gw[j] += err * Z[i][j]
                gb += err
            for j in range(d):
                self.w[j] -= self.lr * (gw[j] / n + self.l2 * self.w[j])
            self.b -= self.lr * (gb / n)
        return self

    def predict_proba(self, x: Sequence[float]) -> float:
        z = self._z(x)
        return _sigmoid(sum(self.w[j] * z[j] for j in range(len(z))) + self.b)
