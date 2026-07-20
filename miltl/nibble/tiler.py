"""Segment tiler — tile a whole call into contiguous, non-overlapping segments (docs/ARCHITECTURE.md).

Text path (no audio): **fixed word-count proxy** (default 14 words ≈ 8 seconds). Covers the
entire call with no gaps.
Audio path (timestamps available): fixed-seconds (8 s) tiling — time_tile (for KsponSpeech/FSS
audio, follow-up).

Note: this tiles the ENTIRE call, not just selected spans. L = ceil(total_words / words_per_seg).
"""
from __future__ import annotations

from typing import List, Optional, Tuple


def tile_by_words(utterances: List[str], words_per_seg: int = 14) -> List[str]:
    """Concatenate utterances in order into a word sequence, then cut into non-overlapping
    chunks of words_per_seg words.

    Returns: list of segment texts (covering the whole call). Empty input → []."""
    toks: List[str] = []
    for u in utterances:
        if u:
            toks.extend(u.split())
    if not toks:
        return []
    return [" ".join(toks[i:i + words_per_seg]) for i in range(0, len(toks), words_per_seg)]


def time_tile(
    timed_utts: List[Tuple[float, float, str]], seconds_per_seg: float = 8.0,
) -> List[str]:
    """Assign (t0,t1,text) utterances to non-overlapping seconds_per_seg-second bins (when audio
    timestamps are available).

    Each utterance goes to the bin of its start time; per-bin texts are concatenated. Tiles the
    entire call (0..max t1)."""
    if not timed_utts:
        return []
    end = max(t1 for _, t1, _ in timed_utts)
    n = int(end // seconds_per_seg) + 1
    bins: List[List[str]] = [[] for _ in range(n)]
    for t0, _t1, text in timed_utts:
        b = min(n - 1, int(t0 // seconds_per_seg))
        if text:
            bins[b].append(text)
    return [" ".join(b) for b in bins]
