"""T/I/F/E channel signal extractors (docs/ARCHITECTURE.md) — nibble → [T,I,F,E] scalars (channel bottleneck).

**Identity (docs/ARCHITECTURE.md)**: each channel = an **independent extractor** (its own inputs, its own supervision). The head sees only this [L,4] (bottleneck).
= structurally distinct from an hf-encoder (embedding→classification). Channels are derived on top of a shared AVD intermediate layer (prosody→A,V,D).

Channel definitions (inherited from docs/ARCHITECTURE.md; extractors are trainable):
  E arousal = A(arousal)                                     input: prosody
  F coercion = f(high D · cold V · rising + directive·threat)  input: AVD + text + speech-act
  I latent  = f(high D · warm V · cross-modal mismatch + subversion)  input: AVD + text + speech-act
  T natural = f(balanced D · prosodic stability + natural conversation)  input: AVD + text

torch. Train the extractors first (D-C) → verify channel discriminative power (F(fss)≫F(emotion)) → head (seq_head). Edge is distilled later.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

CH = ("T", "I", "F", "E")


@dataclass
class ChannelConfig:
    pros_dim: int = 18        # number of ProsodyFeatures features (miltl.nibble.prosody)
    text_dim: int = 768       # klue-roberta mean-pool
    sa_dim: int = 5           # lexical signals: directive/urgency/threat/subversion/warmth (warmth = cross-modal I term)
    d_avd: int = 16           # AVD intermediate hidden size
    d_t: int = 32             # text projection
    hidden: int = 32
    dropout: float = 0.1


def _mlp(sizes, drop=0.1):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers += [nn.SiLU(), nn.Dropout(drop)]
    return nn.Sequential(*layers)


class ChannelExtractors(nn.Module):
    """Nibble features (prosody, text, speech-act) → per-nibble [T,I,F,E] scalars + AVD intermediate representation.

    Each channel head gets a **restricted input** (E = A only, T = AVD+text, F/I = AVD+text+SA) → forces extraction of distinct signals.
    """

    def __init__(self, cfg: ChannelConfig = None):
        super().__init__()
        self.cfg = cfg or ChannelConfig()
        c = self.cfg
        self.avd_net = _mlp([c.pros_dim, c.d_avd, 3], c.dropout)          # shared: prosody→(A,V,D)
        self.text_proj = nn.Sequential(nn.Linear(c.text_dim, c.d_t), nn.SiLU())
        self.F_head = _mlp([3 + c.d_t + c.sa_dim, c.hidden, 1], c.dropout)  # coercion (cold, scam, threat)
        self.I_head = _mlp([3 + c.d_t + c.sa_dim, c.hidden, 1], c.dropout)  # latent manipulation (XM, warmth)
        self.T_head = _mlp([3 + c.d_t + c.sa_dim, c.hidden, 1], c.dropout)  # natural/safe (balanced D, warmth, ¬coercion)
        self.E_head = _mlp([1, 4, 1], c.dropout)                            # arousal (A-based correction)

    def forward(self, pros: torch.Tensor, text: torch.Tensor, sa: torch.Tensor):
        # pros[B,L,pros_dim] · text[B,L,text_dim] · sa[B,L,sa_dim]
        avd = torch.sigmoid(self.avd_net(pros))                 # [B,L,3] = (A,V,D)
        A = avd[..., 0:1]
        t = self.text_proj(text)                                # [B,L,d_t]
        base = torch.cat([avd, t, sa], dim=-1)
        Fc = torch.sigmoid(self.F_head(base)).squeeze(-1)       # coercion
        Ic = torch.sigmoid(self.I_head(base)).squeeze(-1)       # latent
        Tc = torch.sigmoid(self.T_head(base)).squeeze(-1)       # natural (avd+text+lexical)
        Ec = torch.sigmoid(self.E_head(A)).squeeze(-1)          # arousal (A correction)
        tife = torch.stack([Tc, Ic, Fc, Ec], dim=-1)            # [B,L,4] order T,I,F,E
        return tife, avd

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def _selftest() -> int:
    torch.manual_seed(0)
    m = ChannelExtractors()
    B, L = 4, 26
    pros = torch.randn(B, L, 18); text = torch.randn(B, L, 768); sa = torch.rand(B, L, 5)
    tife, avd = m(pros, text, sa)
    assert tife.shape == (B, L, 4) and avd.shape == (B, L, 3), (tife.shape, avd.shape)
    print(f"[selftest] 추출기 forward OK · [B,L,4]={tuple(tife.shape)} · AVD={tuple(avd.shape)} · params={m.num_params():,}")

    # Supervised-learning smoke test: can F be trained high on 'coercive inputs' (verify discriminative power)
    # Synthetic: high sa[threat] → high F target. After training, F(high threat) > F(low threat).
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    for _ in range(150):
        pros = torch.randn(B, L, 18); text = torch.randn(B, L, 768)
        sa = torch.rand(B, L, 5)
        F_tgt = sa[..., 2]                                      # use the threat axis as the F target
        tife, _ = m(pros, text, sa)
        loss = F.mse_loss(tife[..., 2], F_tgt)                 # F channel = index 2
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        sa_hi = torch.zeros(1, 1, 5); sa_hi[..., 2] = 0.9      # high threat
        sa_lo = torch.zeros(1, 1, 5); sa_lo[..., 2] = 0.1
        z = torch.zeros(1, 1, 18); zt = torch.zeros(1, 1, 768)
        F_hi = float(m(z, zt, sa_hi)[0][..., 2]); F_lo = float(m(z, zt, sa_lo)[0][..., 2])
    print(f"[selftest] 감독학습 후 F(高threat)={F_hi:.3f} > F(低threat)={F_lo:.3f} · loss={float(loss):.4f}")
    assert F_hi > F_lo, (F_hi, F_lo)
    print("[selftest] 채널 추출기 동작 — 채널이 자기 신호를 학습·판별. 실감독=SER/speech-act(step2).")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
