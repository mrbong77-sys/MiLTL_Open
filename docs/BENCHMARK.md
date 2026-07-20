# KorMMP — Korean Multi-modal Phishing Benchmark

> Design reference for the lexically-decorrelated hard benchmark and the canonical
> evaluation protocol.

## 1. Thesis

Standard Korean vishing benchmarks reward **vocabulary memorization**: on the public
transcript corpus KorCCViD, frozen legacy text classifiers reach near-perfect scores, which
says nothing about detecting a live scam call whose wording avoids known scam vocabulary.
KorMMP is built to **cut the lexical–harm correlation**: when scam-word density no longer
predicts the label, text-only detectors must collapse, and only signals that survive
decorrelation (prosody trajectories, cross-modal contradiction XM) can separate the classes.

## 2. Slices (density-based, never label-based)

Slicing uses observable lexical density (scam/threat/directive/urgency keyword hits per
word — `scripts/compose_hard_kormmp.py::_density`), never the harm label, to avoid
circularity.

| Slice | Construction | Expected legacy behaviour |
|---|---|---|
| easy-harm | high-density real FSS phishing | detected by everyone (control) |
| hard-harm | ① naturally low-density FSS ② ASR-degraded FSS (transcript corrupted, audio intact) | legacy FN |
| synth-hard-harm | synthetic phishing that **fully avoids** scam vocabulary (grooming, oblique authority pressure), paired with cold real-FSS prosody | legacy FN |
| easy-benign | low-density everyday conversation | trivially correct |
| hard-benign | high-density real benign calls (bank/insurance/call-center consultations) | legacy FP |
| synth-hard-benign | synthetic benign that **deliberately contains** finance/authority vocabulary, paired with warm real-benign prosody (`--cold-benign` adds cold real benign audio as the hardest negative) | legacy FP |

Synthetic scenario diversity: 8 harm archetypes (institution / loan / family / refund /
delivery / acquaintance / investment-coaching / subsidy impersonation) × 8 benign archetypes
(bank, card, insurance, telecom, delivery, government office, hospital, securities) —
`scripts/synth_edgecases.py`, deterministic under a fixed seed. The full Korean scenario
banks are shipped verbatim in this repo, and the exact per-seed bundles used by the bench
are committed at `artifacts/rounds/canonical/synth_*.jsonl`.

## 3. Source-leakage control (audio fairness)

Three measures prevent the audio channel from becoming a corpus-identity shortcut:

1. **Real FSS prosody**: hard slices are materialized from real FSS phishing audio
   (`scripts/materialize_kormmp.py`), not prosody transplantation. Transplantation survives
   only as a source-ablation mode (`compose --transplant`).
2. **Channel equalization**: both classes pass through telephone-band (300–3400 Hz) +
   μ-law codec conditioning (`miltl/nibble/audio_decode.py::equalize_channel`).
3. **Provenance audit**: `scripts/audit_audio_provenance.py` trains a shortcut probe that
   tries to predict the *source corpus* from audio features; the bench is accepted only if
   the shortcut AUROC stays low (the audit exposed a legacy audio baseline whose apparent
   0.90 AUROC was a corpus-channel shortcut — it collapsed to chance after equalization).

## 4. Fairness controls (self-evidencing)

The claim "MiLTL wins because of cross-modal architecture, not modality access" is proven
inside the result sheet itself by two built-in control detectors:

- **Wave-Seq** (`adapters/baselines/wave_seq.py`): audio-only prosody CNN. If audio access
  alone explained the gap, this would survive decorrelation. It does not.
- **MiLTL-Dual**: naive text ⊕ wave score fusion (normalized max-OR) *without* XM. If naive
  fusion sufficed, this would survive. It does not.

Every result sheet carries a `modality` column so the comparison table can be grouped by
modality access.

## 5. Freeze protocol (strict)

- **Training** (calibration, optional head, Gate-2 SFT): KorCCViD-train (real) + a
  *separate* synthetic bundle (seed 99).
- **Evaluation**: frozen KorMMP (seed 42 composition; bench seeds drawn and recorded per
  run). Synthetic-for-training ≠ synthetic-for-evaluation.
- KorMMP is never used for training or tuning. If MiLTL loses, the loss is reported.
- Legacy baselines are frozen KorCCViD-trained checkpoints (see docs/BASELINES.md).

## 6. Canonical orchestrator

`scripts/canonical_bench.py` is the single entry point:

1. draws N random seeds (recorded to `seeds.json`; reproduce with `--seeds`),
2. per corpus × seed: builds one bundle (KorCCViD pool sample / KorMMP synth + compose +
   materialized real pool),
3. runs **all** detectors (legacy + fairness controls + MiLTL-Cascade) in one pass,
4. consolidates a journal table (`consolidated_results.csv`: AUROC/F1/ACC/SEN/SPE/PPV/NPV
   + DeLong tests) via `scripts/consolidate_results.py`.

## 7. Result-sheet schema (faithfulness)

Per call and per detector, the sheet records channels and the decision path:

```
sheet_{corpus}_{seed}.csv:
  detector, modality, call_id, slice, scenario, label, score, pred, outcome,
  T, I, F, E, XM, V, cold, warmth, n_words, density, p1, band, decision [, transcript]
sheet_{corpus}_{seed}.l2ledger.jsonl:   # Gate-2 judgment ledger (audit trail)
  call_id, channel summary, prompt, P(harm), final score, XAI rationale
```

Sheet ↔ ledger cross-validation must show zero mismatches (verified by
`tests/test_consolidate.py` logic). The canonical result files will be published under
`artifacts/rounds/canonical/` when the final benchmark run completes (docs/RESULTS.md).

Note on the shipped manifests: rows derived from AI-Hub corpora carry pointer fields
instead of inline transcripts (see docs/DATA_ACCESS.md and DATA_LICENSE.md).

## 8. Metrics

- **Decorrelated AUROC** on hard slices is the headline metric.
- FPR is tracked specifically on cold/professional benign calls (call-center hard negatives).
- Point metrics use thresholds selected on training data only.
- Multi-seed runs report mean ± std with DeLong significance tests between detectors.
