"""BenchmarkCall — per-call contract of the frozen benchmark (docs/BASELINES.md).

Every detector sees the same call set and the same split. The split is determined by
hashing the call_id (fixed, reproducible — no Date/random, pure hash). raw
(transcript/audio) stays DGX-local; the shared bundle carries metadata only.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from miltl.nibble import load_calls_jsonl, LABEL_MAP


def split_of(call_id: str, test_frac: float = 0.5, salt: str = "miltl-bench-v1") -> str:
    """call_id hash → fixed train/test assignment. The same id always gets the same split (reproducible)."""
    h = hashlib.sha256(f"{salt}:{call_id}".encode("utf-8")).hexdigest()
    frac = int(h[:8], 16) / 0xFFFFFFFF
    return "test" if frac < test_frac else "train"


@dataclass
class BenchmarkCall:
    call_id: str
    label: int                              # 0=benign · 1=phishing
    source: str
    split: str                              # "train" | "test"
    transcript: Optional[str] = None        # for text/LLM families (not pushed)
    audio_uri: Optional[str] = None         # for audio families (not pushed)
    stream: object = None                   # precomputed MiLTL CallStream (for our method)
    meta: Dict = field(default_factory=dict)

    def has(self, need: str) -> bool:
        if need == "text":
            return bool(self.transcript)
        if need == "audio":
            return bool(self.audio_uri)
        if need == "stream":
            return self.stream is not None
        if need == "nibble":                # MiLTL: nibble source = precomputed stream OR featurizable transcript
            return self.stream is not None or bool(self.transcript)
        return False

    def consumable(self, needs) -> bool:
        return all(self.has(n) for n in needs)


def clip_call(call: "BenchmarkCall", max_segments: int, words_per_seg: int = 14) -> "BenchmarkCall":
    """Length control (docs/BENCHMARK.md) — clip the call to its first N segments (≈2 min). Identical input for all detectors (fairness):
    transcript→first N×words_per_seg words, stream→first N segments. max_segments=0 means no change."""
    import dataclasses
    if not max_segments:
        return call
    tr = call.transcript
    if tr:
        tr = " ".join(tr.split()[: max_segments * words_per_seg])
    st = call.stream
    if st is not None and len(getattr(st, "segments", [])) > max_segments:
        from miltl.nibble.schema import CallStream
        st = CallStream(st.call_id, st.source, st.label, st.segments[:max_segments])
    return dataclasses.replace(call, transcript=tr, stream=st)


def benchmark_from_streams(
    phishing_jsonl: str,
    benign_jsonls: List[str],
    test_frac: float = 0.5,
    limit: Optional[int] = None,
) -> List[BenchmarkCall]:
    """Existing nibble streams → BenchmarkCall list (modality=stream). For harness validation and MiLTL self-baselines.

    No raw transcript/audio (streams carry nibbles only) → text/audio-family detectors get skipped.
    External SOTA uses the raw bundle from build_benchmark.py.
    """
    calls: List[BenchmarkCall] = []
    specs = [(phishing_jsonl, 1)] + [(p, 0) for p in benign_jsonls]
    for path, _lab in specs:
        streams = load_calls_jsonl(path)
        if limit:
            streams = streams[:limit]
        for cs in streams:
            if cs.label not in LABEL_MAP:
                continue
            cid = f"{cs.source}:{cs.call_id}"
            calls.append(BenchmarkCall(
                call_id=cid, label=LABEL_MAP[cs.label], source=cs.source,
                split=split_of(cid, test_frac), stream=cs, meta=dict(cs.split_keys or {}),
            ))
    return calls


def load_benchmark(path: str) -> List[BenchmarkCall]:
    """Load a benchmark bundle jsonl. Each line = one call (meta + optional raw + optional stream path).

    Rather than the stream being a reference to a separate nibble jsonl, the bundle itself
    carries the precomputed stream inline as a dict (`stream`) (written alongside by
    build_benchmark.py).
    """
    from miltl.nibble.schema import CallStream
    out: List[BenchmarkCall] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        st = d.get("stream")
        out.append(BenchmarkCall(
            call_id=d["call_id"], label=int(d["label"]), source=d.get("source", "?"),
            split=d.get("split") or split_of(d["call_id"]),
            transcript=d.get("transcript"), audio_uri=d.get("audio_uri"),
            stream=CallStream.from_dict(st) if st else None, meta=d.get("meta", {}),
        ))
    return out
