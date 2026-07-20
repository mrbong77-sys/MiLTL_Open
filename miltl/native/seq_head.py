"""[L,4] sequence decoding head + channel-bottleneck model (docs/ARCHITECTURE.md) — T/I/F/E sequence → harm.

**Bottleneck principle**: the head sees only [L,4] T/I/F/E (no fused representation, no raw embeddings). Discriminative information must pass through the 4 channels.
Head = multi-kernel 1D-CNN (inherits v1 MiLTL-Seq, but with only 4 input channels). Learns trajectories (early I → late F = phishing, sustained T = benign).

ChannelBottleneckModel = ChannelExtractors (docs/ARCHITECTURE.md) + SeqHead. freeze_extractors implements D-C (extractors first → head).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SeqHead(nn.Module):
    """[B,L,4] T/I/F/E sequence → harm logit. Multi-kernel 1D-CNN + masked mean pooling."""

    def __init__(self, in_ch: int = 4, kernels=(3, 5, 7), hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(in_ch, hidden, k, padding=k // 2) for k in kernels])
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Sequential(nn.Linear(hidden * len(kernels), hidden), nn.SiLU(),
                                nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, tife: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # tife[B,L,4] → [B,4,L]
        x = tife.transpose(1, 2)
        feats = [self.act(c(x)) for c in self.convs]            # each [B,hidden,L]
        h = torch.cat(feats, dim=1).transpose(1, 2)             # [B,L,hidden*k]
        if mask is not None:
            m = (mask > 0.5).float().unsqueeze(-1)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)     # masked mean (valid nibbles)
        else:
            pooled = h.mean(1)
        return self.fc(self.drop(pooled)).squeeze(-1)           # [B]


class TSMixerTiny(nn.Module):
    """Ultra-light TSMixer head (docs/ARCHITECTURE.md) — TTM backbone (time-mixing + channel-mixing MLPs), no pretraining.

    Explicitly models the channel interactions of neutrosophic decoding (F+I−T) via the **channel-mixing MLP** (advantage over CNN).
    [B,L,4] → N×(time-mix ⊕ chan-mix) → masked mean pooling (+ dynamics features) → harm. ~15K params.
    """

    def __init__(self, L: int = 26, in_ch: int = 4, n_blocks: int = 2, time_hidden: int = 32,
                 chan_hidden: int = 16, dropout: float = 0.1, use_dyn: bool = True):
        super().__init__()
        self.L, self.C, self.use_dyn = L, in_ch, use_dyn
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.ModuleDict({
                "tn": nn.LayerNorm(in_ch),
                "time": nn.Sequential(nn.Linear(L, time_hidden), nn.SiLU(), nn.Dropout(dropout),
                                      nn.Linear(time_hidden, L)),
                "cn": nn.LayerNorm(in_ch),
                "chan": nn.Sequential(nn.Linear(in_ch, chan_hidden), nn.SiLU(), nn.Dropout(dropout),
                                      nn.Linear(chan_hidden, in_ch)),
            }))
        feat = in_ch + (3 * in_ch if use_dyn else 0)     # pooled + (mean,early,slope) per ch
        self.fc = nn.Sequential(nn.Linear(feat, 32), nn.SiLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def _dyn(self, tife, m):
        # Dynamics features (prior knowledge): per-channel mean, early third, slope. m[B,L].
        B, L, C = tife.shape
        w = m.unsqueeze(-1)
        mean = (tife * w).sum(1) / w.sum(1).clamp(min=1)                  # [B,C]
        k = max(1, L // 3)
        early = (tife[:, :k] * w[:, :k]).sum(1) / w[:, :k].sum(1).clamp(min=1)
        t = torch.linspace(-1, 1, L, device=tife.device).view(1, L, 1)
        slope = ((tife - mean.unsqueeze(1)) * t * w).sum(1) / (t * t * w).sum(1).clamp(min=1e-3)
        return torch.cat([mean, early, slope], dim=-1)                    # [B,3C]

    def forward(self, tife: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        m = (mask > 0.5).float() if mask is not None else torch.ones(tife.shape[:2], device=tife.device)
        x = tife
        for b in self.blocks:
            h = b["tn"](x).transpose(1, 2)                               # [B,C,L]
            x = x + b["time"](h).transpose(1, 2)                         # time-mixing residual
            x = x + b["chan"](b["cn"](x))                                # channel-mixing residual
        w = m.unsqueeze(-1)
        pooled = (x * w).sum(1) / w.sum(1).clamp(min=1)                  # [B,C]
        if self.use_dyn:
            pooled = torch.cat([pooled, self._dyn(tife, m)], dim=-1)
        return self.fc(pooled).squeeze(-1)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class ChannelBottleneckModel(nn.Module):
    """Full model: (prosody, text, speech-act) → extractors → [L,4] → head → harm. MiLTL's identity structure.

    forward returns harm_logit + tife (channels) + avd (for diagnostics/supervision). freeze_extractors=True → train head only (D-C).
    """

    def __init__(self, chan_cfg=None, kernels=(3, 5, 7), hidden: int = 32):
        super().__init__()
        from miltl.native.channels import ChannelExtractors, ChannelConfig
        self.extractors = ChannelExtractors(chan_cfg or ChannelConfig())
        self.head = SeqHead(in_ch=4, kernels=kernels, hidden=hidden)

    def freeze_extractors(self, freeze: bool = True):
        for p in self.extractors.parameters():
            p.requires_grad = not freeze

    def forward(self, pros, text, sa, mask=None):
        tife, avd = self.extractors(pros, text, sa)             # [B,L,4], [B,L,3]
        harm = self.head(tife, mask)
        return {"harm_logit": harm, "tife": tife, "avd": avd}

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def _selftest() -> int:
    torch.manual_seed(0)
    m = ChannelBottleneckModel()
    B, L = 4, 26
    pros = torch.randn(B, L, 18); text = torch.randn(B, L, 768); sa = torch.rand(B, L, 5)
    mask = torch.ones(B, L); mask[0, 20:] = 0
    out = m(pros, text, sa, mask)
    assert out["harm_logit"].shape == (B,) and out["tife"].shape == (B, L, 4)
    print(f"[selftest] 병목모델 forward OK · harm{tuple(out['harm_logit'].shape)} · tife{tuple(out['tife'].shape)} "
          f"· params={m.num_params():,}")
    # Can the head learn harm from the [L,4] sequence: synthetic harm=1 when F channel is high late
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    import torch.nn.functional as Fn
    for _ in range(120):
        tife = torch.rand(B, L, 4)
        y = (tife[:, 15:, 2].mean(1) > 0.5).float()            # harm when late F mean is high
        logit = m.head(tife, torch.ones(B, L))
        loss = Fn.binary_cross_entropy_with_logits(logit, y)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        hi = torch.rand(2, L, 4); hi[:, 15:, 2] = 0.9          # late F high
        lo = torch.rand(2, L, 4); lo[:, 15:, 2] = 0.1
        p_hi = torch.sigmoid(m.head(hi, torch.ones(2, L))).mean()
        p_lo = torch.sigmoid(m.head(lo, torch.ones(2, L))).mean()
    print(f"[selftest] 헤드 궤적학습: harm(F후반高)={float(p_hi):.3f} > harm(F후반低)={float(p_lo):.3f}")
    assert p_hi > p_lo
    print("[selftest] 채널병목(추출기→[L,4]→헤드) 동작 — 헤드가 채널 궤적으로 harm 디코딩.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
