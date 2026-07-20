"""Native multimodal neutrosophic channel core (see docs/ARCHITECTURE.md).

Channel bottleneck design: audio (PCM, telephone band) + transcript are featurized per
nibble, calibrated into the five analytic channels [L,5] = (T, I, F, E, XM), and scored by
the canonical interpretable rule ``risk = E - 2*T + I + XM`` (optionally blended with a
tiny TSMixer head). The head only ever sees the 5 channels — never raw text embeddings.

Only ``channels.py`` (training-time extractors) depends on torch; the analytic inference
path (``channel_calib``, ``nibble_features``) is numpy-only.
"""
