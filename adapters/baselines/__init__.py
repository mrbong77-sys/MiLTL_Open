"""Baseline detector adapters (see docs/BASELINES.md) — each wraps one model as a
``miltl.baseline.BaselineDetector``.

Heavy DL dependencies (torch/transformers/fasttext) are imported lazily inside each
module, so importing this package stays stdlib-safe. Install
``adapters/baselines/requirements.txt`` to run the GPU-backed baselines.

Registered adapters shipped in this repo:
  MiLTL:   native_channel (Gate-1 / full cascade) · gate2_slm (Gate-2 SLM arbiter)
  Text:    hf_encoder · cnn_bilstm_fasttext · tree_ensemble · miltl_seq (legacy text line)
  LLM:     bllossom_repro (B3 original-method reproduction) · bllossom_llm (prompt bank)
  Audio:   wave_seq (audio-only fairness control; also the naive dual-fusion control)
"""
