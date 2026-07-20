# Architecture — Channel-Bottleneck Cascade

> Canonical design reference for MiLTL (Multimodal In-call Lightweight Threat Locator).
> Consolidated from the private research repo's canonical-state documents.

## 1. One-paragraph summary

MiLTL is a two-stage cascade for on-device vishing detection. **Gate-1** (always-on,
lightweight, recall-first) encodes each 8-second segment ("nibble") of a call into five
neutrosophic/affect channels and scores risk with a deterministic interpretable rule.
Calls whose Gate-1 probability lands in the uncertainty band escalate to **Gate-2**, a small
LLM arbiter that reads the transcript plus an XM channel summary and issues the final
verdict with an explainable rationale and recommended edge actions. The decisive signal is
**XM (cross-modal contradiction)** — warm/cooperative *text* delivered with cold/flat
*prosody* — which single-modality detectors structurally cannot observe.

## 2. Inference path

```
audio (.wav/.mp4) ──decode──▶ PCM (16 kHz, telephone band 300–3400 Hz)
transcript (ASR) ─┐
                  ▼
  featurize_channels()  →  NibbleChannelInput (prosody[L,18] · speech_act[L,4] · warmth[L] · mask)   L = 26
                  ▼
  channel_calib.channels5(nci, Calib) → [L,5] = (T, I, F, E, XM)     ← analytic membership functions
                  ▼                                                     (calibrated logistics; no text-768 dependency)
  ★ Gate-1 canonical scorer (deterministic):  risk = E − 2·T + I + XM
     · observation anchor: with audio ≥80 words, transcript-only ≥12 words; below = undecidable (0.0)
     · no audio (transcript-only, e.g. KorCCViD): prosodic E/XM are inert → lexical-F recentring
       compensation  risk += 2·(F − 0.5)
     · (optional) learned TSMixer head mixed only via blend_analytic < 1 — the analytic rule is canonical
                  ▼
  p1 ∈ [0,1]  →  banding:   p1 ≤ 0.40 → benign (close) · p1 ≥ 0.90 → harm (early alarm)
                            0.40 < p1 < 0.90 → escalate to Gate-2
                  ▼
  ★ Gate-2 (escalated calls only): Gate2SLM (Qwen2.5-1.5B + LoRA SFT)
     · prompt = transcript (primary) + XM channel summary injection (Korean prompt, functional data)
     · P(harm) mapped back into the band:  final = τ_low + P·(τ_high − τ_low)  → globally monotone with p1
                  ▼
  XAI: explain_decision(channels, verdict, transcript, p1) → reasons · recommended action
       (warn / end call / notify guardian / report to 112 or 1332)
```

**Canonical choices.** Gate-1 = the deterministic rule `E − 2T + I + XM` (0 learned
parameters, interpretable, edge-native). The trained head and the torch channel extractors
are optional/auxiliary — the extractor checkpoint is used only to supply calibration
statistics (`Calib`). The canonical Gate-2 backbone is **Qwen2.5-1.5B-Instruct**
(0.5B is an edge-lightweight ablation).

## 3. Channel definitions (neutrosophic SVNS + affect VAD)

Implemented in `miltl/native/channel_calib.py`. Evidence terms are soft-OR combinations
passed through calibrated logistics; calibration statistics come from benign-train prosody
(median/IQR z-normalization, channel thresholds = benign percentiles), frozen on
KorCCViD-train benign only.

| Channel | Meaning | Evidence (soft-OR, calibrated logistic) | Canonical weight |
|---|---|---|---|
| **T** (truth) | natural / cooperative conversation (benign) | coherence + 0.30·(1−XM) + 0.20·balancedD + 0.15·warmth − 0.5·coercion | **−2** (strongest channel) |
| **E** (energy) | arousal (PAD threat signature) | = A (arousal) | **+1** (second strongest) |
| **I** (indeterminacy) | covert manipulation (grooming) | 0.85·XM + 0.15·(warmth·has_ask) | **+1** |
| **XM** (cross-modal) | **novelty signal** — warm text vs. cold prosody | clip(warmth_text − V_prosody)·(0.5 + 0.5·D) | **+1** (invisible to legacy detectors) |
| **F** (falsity) | overt coercion | 0.40·cold + 0.30·subversion + … | **excluded** when audio present (no discriminative signal on this distribution; main FP driver on cold call-center benigns). Used only as the lexical recentring term in transcript-only mode. |

The weights were derived from a 5-seed, 500-call channel-discriminability analysis on the
design pool (T strongest, E second, F no signal with audio present) and then frozen; see
docs/BENCHMARK.md for the freeze protocol.

## 4. Operating points

| Parameter | Value | Where |
|---|---|---|
| Nibble length | 8 s (≈14 words transcript proxy) | `configs/nibble.yaml` (`stream.segment_seconds`) |
| Observation budget | first 26 nibbles (≈360 words) | `miltl/native/features.py` (`MAX_NIBBLES`) |
| Banding thresholds | τ_low = 0.40, τ_high = 0.90 | `adapters/baselines/native_channel.py`, `scripts/gate2_cascade.py` |
| Observation anchors | 80 words (audio) / 12 words (transcript-only) | `native_channel` (`anchor_words`) |
| Canonical scorer | `risk = E − 2·T + I + XM`, `blend_analytic = 1.0` | `native_channel` |
| Audio conditioning | telephone band 300–3400 Hz + μ-law codec equalization | `miltl/nibble/audio_decode.py` |
| Calibration freeze | KorCCViD-train benign only | `channel_calib.fit_calib` |

`configs/nibble.yaml` additionally records the historical threshold-calibration lineage
(seed → calibrated_v1 → derived_v2 → derived_wave_tb) with the rationale for each step.

## 5. Why a channel bottleneck?

The head (analytic rule or optional TSMixer) sees **only** the [L,5] channel sequence —
never raw text embeddings. This is a structural difference from encoder classifiers
(embedding → classification), and it is what makes the lexical-decorrelation benchmark
meaningful: a detector that cannot see vocabulary cannot win by memorizing scam vocabulary.
It also gives XAI for free — every decision decomposes into named, bounded channel
contributions (`miltl/native/explain.py`).

## 6. Components

| File | Role |
|---|---|
| `miltl/native/channel_calib.py` | canonical channels: `Calib`, `evidence` (zF/zI/zT/XM), `channels5` → [L,5] |
| `miltl/native/nibble_features.py` | `featurize_channels()` → NibbleChannelInput |
| `miltl/native/explain.py` | XAI: `explain_decision(diag, decision, transcript, p1)` → verdict/reasons/action/summary |
| `miltl/native/seq_head.py` | `TSMixerTiny(in_ch=5)` optional head (≈4.5K params) |
| `miltl/native/channels.py` | torch channel extractors (training-time; checkpoint supplies `Calib` only) |
| `miltl/native/channel_teacher.py` | Korean lexicon banks (functional data — kept verbatim) |
| `adapters/baselines/native_channel.py` | MiLTL Gate-1 detector (`ChannelBottleneckDetector`), banding, diagnostics |
| `adapters/baselines/gate2_slm.py` | Gate-2 SLM: `judge_channels`, `fit_channels` (LoRA SFT), XM-injected prompt |
| `adapters/baselines/tqa.py` | Threat-Question-Answering bank for situation-conditioned Gate-2 prompting |
