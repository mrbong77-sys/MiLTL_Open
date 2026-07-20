"""MiLTL — Multimodal In-call Lightweight Threat Locator.

Edge-side Korean voice-phishing (vishing) detector: audio + transcript are observed in
8-second nibbles, encoded into neutrosophic/affect channels [L,5] = (T, I, F, E, XM), and
decoded to a harm decision by a lightweight head (Gate-1) with a small-LLM arbiter (Gate-2).
"""
