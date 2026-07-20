"""DGX integration entry point — inject real featurizers + assemble real-path CallStream (see design notes).

DGX wraps its own PEINN/acoustic heads in thin adapter functions and injects them via a
`module:function` spec. This module orchestrates tiling → featurize → build_call_stream/attach_wave
(reusing the existing pieces).

Here (dev environment) we verify the assembly runs with Mock text + ProsodyWaveFeaturizer.
"""
from __future__ import annotations

import importlib
from typing import Callable, List, Optional

from .schema import CallStream, build_call_stream, attach_wave
from .tiler import tile_by_words
from .tife import TIFE


def load_callable(spec: str) -> Callable:
    """"package.module:function" → load a callable. For DGX adapter injection.

    Example: --text-adapter myproj.peinn_adapter:score  (score(text)->(T,I,F,E))
    """
    if ":" not in spec:
        raise ValueError(f"어댑터 스펙은 'module:function' 형식이어야 함: {spec!r}")
    mod, fn = spec.split(":", 1)
    try:
        m = importlib.import_module(mod)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"어댑터 모듈 '{mod}' 을 찾을 수 없음 ({e}). 확인: "
            f"(1) 어댑터 .py 파일을 실제로 만들었는가, "
            f"(2) 그 위치가 sys.path 에 있는가(MiLTL repo 안이거나 PYTHONPATH), "
            f"(3) 어댑터가 PEINN 을 import 하면 PEAOS 도 PYTHONPATH 에 있는가."
        ) from e
    if not hasattr(m, fn):
        raise AttributeError(f"'{mod}' 에 함수 '{fn}' 없음. 사용 가능: {[a for a in dir(m) if not a.startswith('_')][:20]}")
    return getattr(m, fn)


def assemble_multimodal_stream(
    call_id: str, source: str, label: Optional[str],
    text_segments: Optional[List[str]] = None,
    text_featurizer=None,
    prosody_dicts: Optional[List[dict]] = None,
    wave_featurizer=None,
    seconds_per_seg: float = 8.0,
    with_transcript: bool = False,
    **meta,
) -> CallStream:
    """Already-tiled text segments / prosody segments → multimodal CallStream.

    text_featurizer.featurize_many(text_segments) → text_tife stream.
    wave_featurizer.featurize_many(prosody_dicts) → wave_tife stream.
    The two streams are aligned by segment idx (up to min length). Time alignment is the
    caller's responsibility (recommended: 8-second tiling for both).
    """
    text_tifes: Optional[List[Optional[TIFE]]] = None
    wave_tifes: Optional[List[Optional[TIFE]]] = None
    if text_segments is not None and text_featurizer is not None:
        text_tifes = list(text_featurizer.featurize_many(text_segments))
    if prosody_dicts is not None and wave_featurizer is not None:
        wave_tifes = list(wave_featurizer.featurize_many(prosody_dicts))

    cs = build_call_stream(
        call_id=call_id, source=source, label=label,
        text_tifes=text_tifes, wave_tifes=wave_tifes,
        seconds_per_seg=seconds_per_seg,
        transcripts=(text_segments if with_transcript else None),
        **meta,
    )
    return cs


def tile_text(utterances: List[str], words_per_seg: int = 14) -> List[str]:
    """Convenience re-export — text word-count-proxy tiling."""
    return tile_by_words(utterances, words_per_seg)
