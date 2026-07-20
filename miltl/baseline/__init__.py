"""Baseline benchmark harness (see docs/BASELINES.md) — controlled SOTA comparison.

MiLTL and external detectors (local LLM, text encoders, audio) are compared on the same
benchmark, same metrics, and same result-sheet format. The core harness is pure stdlib;
heavy external models are isolated in ``adapters/baselines/``.
"""
from .bench import BenchmarkCall, load_benchmark, benchmark_from_streams, split_of
from .detector import BaselineDetector, ResultRow, compute_metrics, run_benchmark
from .sheet import render_sheet, write_sheet, render_matrix

__all__ = [
    "BenchmarkCall", "load_benchmark", "benchmark_from_streams", "split_of",
    "BaselineDetector", "ResultRow", "compute_metrics", "run_benchmark",
    "render_sheet", "write_sheet", "render_matrix",
]
