"""CallStream consumption layer — jsonl loading + multimodal window construction (docs/ARCHITECTURE.md).

Adapter that carries the contract (CallStream) all the way through the training side (Gate-1).
Windows are segment slices and inherit the call label (no new labeling; see docs/ARCHITECTURE.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .schema import CallStream

LABEL_MAP = {"benign": 0, "phishing": 1}


@dataclass
class MMWindow:
    text_nibbles: List[Optional[int]]
    wave_nibbles: List[Optional[int]]
    label: int
    call_id: str = ""


def load_calls_jsonl(path: str) -> List[CallStream]:
    return [CallStream.from_jsonl_line(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def windows_from_calls(
    calls: List[CallStream], win: int = 15, stride: int = 4, min_len: int = 6,
) -> List[MMWindow]:
    """Slide a win-segment (~2 min) window over CallStreams. Windows inherit the call label (unlabeled calls are skipped)."""
    out: List[MMWindow] = []
    for c in calls:
        if c.label not in LABEL_MAP:
            continue
        y = LABEL_MAP[c.label]
        tn, wn = c.nibble_streams()
        L = len(tn)
        if L <= win:
            if L >= min_len:
                out.append(MMWindow(tn, wn, y, c.call_id))
            continue
        for s in range(0, L - win + 1, stride):
            out.append(MMWindow(tn[s:s + win], wn[s:s + win], y, c.call_id))
    return out
