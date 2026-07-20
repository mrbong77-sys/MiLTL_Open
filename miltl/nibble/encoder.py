"""Nibble encoder — TIFE -> 4-bit nibble, plus a streaming accumulator.

Bit layout (MSB->LSB):  [T][I][F][E]
    value = 8*bT + 4*bI + 2*bF + 1*bE   in {0..15}

Thresholds are CALIBRATION artifacts, not fixed constants (docs/ARCHITECTURE.md). This module
carries seed defaults, but in real use the validation-recalibrated tau values are injected.
E is the MiLTL prosody channel, so it arrives self-normalized to [0,1] with no [0,10] assumption
(see tife.TIFE.E).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .tife import TIFE

# Bit positions
T_BIT = 3
I_BIT = 2
F_BIT = 1
E_BIT = 0


@dataclass
class NibbleThresholds:
    """Independent per-channel thresholds. Seeds are the v2.1 text operating point (reference only) — subject to recalibration in MiLTL."""
    tau_T: float = 0.75      # v2.1 tau_safe (seed)
    tau_F: float = 0.50      # v2.1 tau_harm (seed)
    tau_I: float = 0.65      # v2.1 tau_I    (seed)
    tau_C: float = 0.30      # v2.1 tau_C, C=min(T,F) (seed)
    tau_E: float = 0.50      # MiLTL prosody channel — no seed available; must be reset from its own distribution.
    use_conflict: bool = True  # bI = 1[I>=tau_I OR C>=tau_C]

    def is_seed(self) -> bool:
        """Whether still in the pre-calibration seed state (for logging/warnings)."""
        return (self.tau_T, self.tau_F, self.tau_I, self.tau_C) == (0.75, 0.50, 0.65, 0.30)


class NibbleEncoder:
    """TIFE -> nibble (int 0..15). On/off decision per channel with independent thresholds."""

    def __init__(self, thresholds: NibbleThresholds | None = None):
        self.th = thresholds or NibbleThresholds()

    def bits(self, x: TIFE) -> tuple[int, int, int, int]:
        th = self.th
        bT = 1 if x.T >= th.tau_T else 0
        bF = 1 if x.F >= th.tau_F else 0
        bE = 1 if x.E >= th.tau_E else 0
        bI = 1 if x.I >= th.tau_I else 0
        if th.use_conflict and x.C >= th.tau_C:   # uncertainty = ignorance OR contradiction
            bI = 1
        return bT, bI, bF, bE

    def encode(self, x: TIFE) -> int:
        bT, bI, bF, bE = self.bits(x)
        return (bT << T_BIT) | (bI << I_BIT) | (bF << F_BIT) | (bE << E_BIT)

    def encode_stream(self, xs: Iterable[TIFE]) -> List[int]:
        return [self.encode(x) for x in xs]


def unpack(nib: int) -> tuple[int, int, int, int]:
    """nibble -> (bT, bI, bF, bE)."""
    return (
        (nib >> T_BIT) & 1,
        (nib >> I_BIT) & 1,
        (nib >> F_BIT) & 1,
        (nib >> E_BIT) & 1,
    )


def pack_bytes(nibbles: List[int]) -> bytes:
    """4-bit packing: 2 nibbles = 1 byte (edge ring-buffer storage format). If odd, the last byte holds only the high nibble."""
    out = bytearray()
    for i in range(0, len(nibbles), 2):
        hi = nibbles[i] & 0xF
        lo = (nibbles[i + 1] & 0xF) if i + 1 < len(nibbles) else 0
        out.append((hi << 4) | lo)
    return bytes(out)


def unpack_bytes(packed: bytes, n: int) -> List[int]:
    """Inverse of pack_bytes. n = original nibble count."""
    out: List[int] = []
    for b in packed:
        out.append((b >> 4) & 0xF)
        out.append(b & 0xF)
    return out[:n]


class NibbleAccumulator:
    """Streaming nibble accumulator — mimics the edge ring buffer (most recent maxlen entries)."""

    def __init__(self, encoder: NibbleEncoder, maxlen: int | None = None):
        self.encoder = encoder
        self.maxlen = maxlen
        self._buf: List[int] = []

    def push(self, x: TIFE) -> int:
        nib = self.encoder.encode(x)
        self._buf.append(nib)
        if self.maxlen is not None and len(self._buf) > self.maxlen:
            self._buf = self._buf[-self.maxlen :]
        return nib

    @property
    def nibbles(self) -> List[int]:
        return list(self._buf)

    def __len__(self) -> int:
        return len(self._buf)
