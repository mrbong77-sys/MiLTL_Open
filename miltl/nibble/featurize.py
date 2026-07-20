"""Text featurizer — segment text → PEINN text-head output TIFE (docs/ARCHITECTURE.md).

Production/DGX: PEINN neutrosophic text head (independent 3-sigmoid T/I/F) + energy → TIFE.
Dev environment: no real PEINN checkpoint is available, so a Mock validates the pipeline
(identical interface).
"""
from __future__ import annotations

from typing import Callable, List, Optional, Protocol, Tuple

from .tife import TIFE, MockTIFEProvider


class TextFeaturizer(Protocol):
    def featurize(self, segment_text: str) -> TIFE: ...

    def featurize_many(self, segments: List[str]) -> List[TIFE]: ...


class MockTextFeaturizer:
    """Smoke-test/demo only — keyword-based mock (reuses tife.MockTIFEProvider). Not the real head."""

    def __init__(self):
        self._p = MockTIFEProvider()

    def featurize(self, segment_text: str) -> TIFE:
        return self._p.segment(segment_text, None)

    def featurize_many(self, segments: List[str]) -> List[TIFE]:
        return [self.featurize(s) for s in segments]


class PeinnTextFeaturizer:
    """Binding seam for the real PEINN text head on DGX.

    Inject a callable score_fn(text) -> (T, I, F, E). On DGX, wrap PEINN's neutro head (T/I/F) +
    energy and pass it in. Without injection, a clear error is raised (use the Mock in dev).

        # DGX example:
        # from peinn_v2.energy import score_axes  # or the neutro head loader
        # def score(text): ...neutro T/I/F + energy... return (T,I,F,E)
        # feats = PeinnTextFeaturizer(score)
    """

    def __init__(self, score_fn: Optional[Callable[[str], Tuple[float, float, float, float]]] = None):
        self._score_fn = score_fn

    def featurize(self, segment_text: str) -> TIFE:
        if self._score_fn is None:
            raise RuntimeError(
                "PeinnTextFeaturizer: score_fn 미주입. DGX 에서 실 PEINN 텍스트 head 를 "
                "score_fn(text)->(T,I,F,E) 로 주입하거나, 개발환경에선 MockTextFeaturizer 를 쓰세요.")
        return TIFE(*self._score_fn(segment_text)).clamp()

    def featurize_many(self, segments: List[str]) -> List[TIFE]:
        return [self.featurize(s) for s in segments]
