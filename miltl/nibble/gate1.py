"""Gate-1 — first-stage nibble pattern discriminator (MiLTL core). Produces p1 = P(harm).

The baseline is interpretable weighted rules + logistic (pure stdlib, no training required,
for cold start). Once a training corpus is available, swap in GBDT / 1D-CNN / GRU behind the
same interface (docs/ARCHITECTURE.md).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .features import NibbleFeatures


class Gate1Scorer(Protocol):
    def score(self, feats: NibbleFeatures) -> float: ...


@dataclass
class RuleLogisticGate1:
    """Weighted rules → logistic. Weights are an interpretable seed (domain prior knowledge) —
    to be replaced by training.

    Strong signals: F persistence, F∧E co-firing (urgency + phishing), harm ramp (escalating trajectory).
    Mitigating signal: T (normal) rate.
    I load is handled by cascade escalation rather than lowering confidence here (small direct penalty).
    """
    w_f_rate: float = 2.2
    w_f_max_run: float = 1.6
    w_fe_cooccur: float = 2.6      # harm + pressure together = strongest phishing signal
    w_harm_ramp: float = 1.2
    w_e_rate: float = 0.5
    w_t_rate: float = -1.8         # normal-confidence → push down
    w_i_load: float = -0.3         # indeterminacy slightly lowers harm confidence (defer decision to cascade)
    bias: float = -1.4

    def logit(self, feats: NibbleFeatures) -> float:
        return (
            self.bias
            + self.w_f_rate * feats.f_rate
            + self.w_f_max_run * feats.f_max_run
            + self.w_fe_cooccur * feats.fe_cooccur
            + self.w_harm_ramp * feats.harm_ramp
            + self.w_e_rate * feats.e_rate
            + self.w_t_rate * feats.t_rate
            + self.w_i_load * feats.i_load
        )

    def score(self, feats: NibbleFeatures) -> float:
        if feats.n_segments == 0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-self.logit(feats)))
