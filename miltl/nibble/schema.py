"""Multimodal nibble data contract (docs/ARCHITECTURE.md) — the pipeline's goalposts.

For each segment, the two paths each produce a neutrosophic [T I F E] -> nibble (4 bits), and
the fused byte = (text_nibble << 4) | wave_nibble. The byte sequence obtained by tiling the
whole call into non-overlapping 8-second segments is the unit of training and decision.

Core principles:
  - The two modalities are **symmetric** (both T/I/F/E). encoder.NibbleEncoder is reused as
    separate instances for text and acoustics.
  - **Missing modalities are allowed**: text-only (everyday conversation) and audio-only corpora
    exist, so a per-modality presence mask is carried.
  - The raw continuous values (raw tife) are optionally stored alongside, for threshold
    recalibration and simulation.
  - Governance: shared artifacts contain no raw audio or PII. Transcripts are stored optionally,
    subject to license/PII policy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

from .tife import TIFE
from .encoder import NibbleEncoder, NibbleThresholds, unpack

SCHEMA_VERSION = "miltl-nibble/0.1"

# Byte layout: high 4 bits = text nibble, low 4 bits = wave (acoustic) nibble.
TEXT_SHIFT = 4
WAVE_MASK = 0x0F


def fuse_byte(text_nibble: Optional[int], wave_nibble: Optional[int]) -> Optional[int]:
    """Two nibbles -> one 8-bit byte. None if either side is missing (missingness expressed via the mask)."""
    if text_nibble is None or wave_nibble is None:
        return None
    return ((text_nibble & 0x0F) << TEXT_SHIFT) | (wave_nibble & 0x0F)


def split_byte(byte: int) -> Tuple[int, int]:
    """Byte -> (text_nibble, wave_nibble)."""
    return (byte >> TEXT_SHIFT) & 0x0F, byte & WAVE_MASK


def _tife_to_list(x: Optional[TIFE]) -> Optional[list]:
    return None if x is None else [round(x.T, 6), round(x.I, 6), round(x.F, 6), round(x.E, 6)]


def _list_to_tife(v) -> Optional[TIFE]:
    return None if v is None else TIFE(*v)


@dataclass
class SegmentRecord:
    """One segment (= one nibble unit; default 8 seconds, or a 14-word proxy for text)."""
    idx: int                                    # segment index (0-based, contiguous within the call)
    t0: Optional[float] = None                  # start second (when audio exists). Text-only: None/word index
    t1: Optional[float] = None                  # end second
    text_nibble: Optional[int] = None           # text-path nibble (0..15) — None if missing
    wave_nibble: Optional[int] = None           # acoustic-path nibble (0..15) — None if missing
    text_tife: Optional[TIFE] = None            # raw text-path [T,I,F,E] (for threshold simulation, optional)
    wave_tife: Optional[TIFE] = None            # raw acoustic-path values (optional)
    transcript: Optional[str] = None            # transcript of this segment (optional per PII policy)

    @property
    def byte(self) -> Optional[int]:
        return fuse_byte(self.text_nibble, self.wave_nibble)

    @property
    def has_text(self) -> bool:
        return self.text_nibble is not None

    @property
    def has_wave(self) -> bool:
        return self.wave_nibble is not None

    def to_dict(self, with_raw: bool = True, with_transcript: bool = False) -> dict:
        d = {"idx": self.idx, "t0": self.t0, "t1": self.t1,
             "text_nibble": self.text_nibble, "wave_nibble": self.wave_nibble,
             "byte": self.byte}
        if with_raw:
            d["text_tife"] = _tife_to_list(self.text_tife)
            d["wave_tife"] = _tife_to_list(self.wave_tife)
        if with_transcript and self.transcript is not None:
            d["transcript"] = self.transcript
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SegmentRecord":
        return cls(
            idx=d["idx"], t0=d.get("t0"), t1=d.get("t1"),
            text_nibble=d.get("text_nibble"), wave_nibble=d.get("wave_nibble"),
            text_tife=_list_to_tife(d.get("text_tife")),
            wave_tife=_list_to_tife(d.get("wave_tife")),
            transcript=d.get("transcript"),
        )


@dataclass
class CallStream:
    """Multimodal nibble stream of one call (or utterance session) + label/meta. Storage unit = one of these objects = one jsonl line."""
    call_id: str
    source: str                                 # "fss" | "ksponspeech" | "dailydialog130" | "synthetic" ...
    label: Optional[str] = None                 # "phishing" | "benign" | None (unlabeled)
    segments: List[SegmentRecord] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    segment_seconds: Optional[float] = 8.0      # tiling parameter (audio)
    segment_words: Optional[int] = 14           # tiling parameter (text proxy)
    sample_rate: Optional[int] = None           # original audio sample rate (when present)
    split_keys: dict = field(default_factory=dict)   # {speaker, source, channel} — for disjoint splits

    # ---- convenience accessors ----
    @property
    def modality(self) -> dict:
        return {"text": any(s.has_text for s in self.segments),
                "wave": any(s.has_wave for s in self.segments)}

    def byte_stream(self) -> List[Optional[int]]:
        """Byte sequence in segment order (None where missing)."""
        return [s.byte for s in self.segments]

    def nibble_streams(self) -> Tuple[List[Optional[int]], List[Optional[int]]]:
        """(text_nibble sequence, wave_nibble sequence) — convenient for per-modality masking in ML."""
        return ([s.text_nibble for s in self.segments], [s.wave_nibble for s in self.segments])

    # ---- serialization ----
    def to_dict(self, with_raw: bool = True, with_transcript: bool = False) -> dict:
        return {
            "schema_version": self.schema_version, "call_id": self.call_id,
            "source": self.source, "label": self.label,
            "segment_seconds": self.segment_seconds, "segment_words": self.segment_words,
            "sample_rate": self.sample_rate, "split_keys": self.split_keys,
            "modality": self.modality,
            "segments": [s.to_dict(with_raw, with_transcript) for s in self.segments],
        }

    def to_jsonl_line(self, **kw) -> str:
        return json.dumps(self.to_dict(**kw), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "CallStream":
        return cls(
            call_id=d["call_id"], source=d["source"], label=d.get("label"),
            segments=[SegmentRecord.from_dict(s) for s in d.get("segments", [])],
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            segment_seconds=d.get("segment_seconds"), segment_words=d.get("segment_words"),
            sample_rate=d.get("sample_rate"), split_keys=d.get("split_keys", {}),
        )

    @classmethod
    def from_jsonl_line(cls, line: str) -> "CallStream":
        return cls.from_dict(json.loads(line))

    # ---- validation ----
    def validate(self) -> List[str]:
        """Return contract violations as a list of strings (empty list = OK)."""
        errs: List[str] = []
        if self.label not in (None, "phishing", "benign"):
            errs.append(f"label 부적합: {self.label!r}")
        if not self.segments:
            errs.append("segments 비어있음")
        for s in self.segments:
            for nm, nb in (("text", s.text_nibble), ("wave", s.wave_nibble)):
                if nb is not None and not (0 <= nb <= 15):
                    errs.append(f"seg{s.idx} {nm}_nibble 범위초과: {nb}")
            if not s.has_text and not s.has_wave:
                errs.append(f"seg{s.idx}: 두 모달 모두 결측")
        idxs = [s.idx for s in self.segments]
        if idxs != list(range(len(idxs))):
            errs.append("segment idx 가 0..N 연속이 아님(타일링 규약 위반)")
        return errs


def build_call_stream(
    call_id: str, source: str, label: Optional[str],
    text_tifes: Optional[List[Optional[TIFE]]] = None,
    wave_tifes: Optional[List[Optional[TIFE]]] = None,
    text_enc: Optional[NibbleEncoder] = None,
    wave_enc: Optional[NibbleEncoder] = None,
    seconds_per_seg: float = 8.0,
    transcripts: Optional[List[Optional[str]]] = None,
    **meta,
) -> CallStream:
    """Per-modality TIFE sequences (one per segment, None where missing) -> CallStream. Both sequences must be the same length.

    text_enc/wave_enc: per-modality NibbleEncoder (each with its own thresholds). Defaults to seed thresholds if omitted.
    transcripts: per-segment transcripts (optional; not stored by default per governance).
    """
    text_enc = text_enc or NibbleEncoder(NibbleThresholds())
    wave_enc = wave_enc or NibbleEncoder(NibbleThresholds())
    n = max(len(text_tifes or []), len(wave_tifes or []))
    segs: List[SegmentRecord] = []
    for i in range(n):
        tt = (text_tifes[i] if text_tifes and i < len(text_tifes) else None)
        wt = (wave_tifes[i] if wave_tifes and i < len(wave_tifes) else None)
        segs.append(SegmentRecord(
            idx=i,
            t0=round(i * seconds_per_seg, 3), t1=round((i + 1) * seconds_per_seg, 3),
            text_nibble=(text_enc.encode(tt) if tt is not None else None),
            wave_nibble=(wave_enc.encode(wt) if wt is not None else None),
            text_tife=tt, wave_tife=wt,
            transcript=(transcripts[i] if transcripts and i < len(transcripts) else None),
        ))
    return CallStream(call_id=call_id, source=source, label=label,
                      segments=segs, segment_seconds=seconds_per_seg, **meta)


def attach_wave(cs: CallStream, wave_tifes: List[Optional[TIFE]],
                wave_enc: Optional[NibbleEncoder] = None) -> CallStream:
    """Attach an acoustic TIFE stream to an existing (text) CallStream, aligned by segment idx -> multimodal.

    Maps wave_tifes[i] onto segment[i] (only the overlapping range if lengths differ). Used to merge
    text-path output and acoustic-path output into a single contract object (e.g. FSS text stream +
    FSS audio prosody)."""
    wave_enc = wave_enc or NibbleEncoder(NibbleThresholds())
    for i, seg in enumerate(cs.segments):
        if i < len(wave_tifes) and wave_tifes[i] is not None:
            seg.wave_tife = wave_tifes[i]
            seg.wave_nibble = wave_enc.encode(wave_tifes[i])
    return cs
