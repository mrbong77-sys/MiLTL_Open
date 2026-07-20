# Results — Canonical Benchmark

The canonical benchmark (`scripts/canonical_bench.py`, **5 random seeds**, all baselines +
MiLTL cascade, one pass per corpus per seed) is complete. The result sheets are published
under [`artifacts/rounds/canonical/`](../artifacts/rounds/canonical/). This is the **final
hard-set run** (2026-07-20 11:17 KST): its KorMMP mix is the fully lexically-decorrelated
distribution (pooled harm mean density 1.80 vs benign 1.21 — an earlier, easier-mix run was
discarded; provenance is logged in the source repo's research log).

- Seeds (recorded, reproducible with `--seeds`): `856444, 715899, 494861, 510553, 439036`
  (`seeds.json`); n = 500 calls per corpus (100 × 5 seeds).
- Gate-2 backbone: **Qwen2.5-1.5B-Instruct** with the released LoRA SFT adapter
  (`artifacts/models/gate2_adapter_1.5b/`, docs/MODELS.md). The release Calib is
  `artifacts/models/calib.release.json`.

## Journal table (`consolidated_results.csv`)

Metrics are AUROC / F1 (Youden-threshold) with sensitivity (SEN) and specificity (SPE)
called out where they explain the failure mode.

### KorMMP — lexically-decorrelated hard benchmark (the meaningful test)

| Group | Detector | AUROC | F1 | SEN | SPE |
|---|---|---|---|---|---|
| **MiLTL (proposed)** | **MiLTL-Cascade** | **0.965** | **0.939** | 0.955 | 0.947 |
| MiLTL (proposed) | MiLTL-L1 (Gate-1 only, ablation) | 0.933 | 0.840 | 0.840 | 0.893 |
| Legacy ML/encoder (text) | cnn-bilstm (frozen) | 0.658 | 0.677 | 0.965 | 0.410 |
| Legacy ML/encoder (text) | hf-encoder (frozen) | 0.641 | 0.701 | 1.000 | 0.430 |
| Legacy ML/encoder (text) | tree (frozen) | 0.638 | 0.520 | 0.515 | 0.690 |
| Legacy ML/encoder (text) | lexical (text proxy) | 0.407 | 0.383 | 0.245 | 0.977 |
| Legacy LLM (text) | Bllossom-B3 (8B, public fine-tuned) | 0.684 | 0.624 | 0.580 | 0.813 |
| Audio-only | Wave-Seq | 0.508 | 0.576 | 1.000 | 0.017 |
| Naive fusion (multimodal) | MiLTL-Dual (text + wave, no XM) | 0.561 | 0.547 | 0.600 | 0.603 |

Reading the failure modes:

- **Text legacy detectors collapse on hard-benign (low SPE).** hf-encoder catches all harm
  (SEN 1.0) but flags 57 % of legitimate calls (SPE 0.43); cnn-bilstm behaves the same way —
  they learned "finance/authority vocabulary = scam" on KorCCViD and over-fire off-corpus.
- **The lexical proxy also misses hard-harm (SEN 0.245):** phishing that avoids scam
  vocabulary is invisible to keyword matching; the frozen tree splits the difference and
  fails in both directions (F1 0.520).
- **Audio-only (Wave-Seq 0.508) and naive fusion (MiLTL-Dual 0.561) both collapse** — so
  MiLTL's margin is the **XM cross-modal architecture**, not mere modality access or having
  audio at all. (Wave-Seq's earlier 0.90 was a provenance shortcut; after channel
  equalization — telephone band + μ-law applied uniformly — it drops to chance.)
- **The 8B Bllossom-B3 LLM (0.684)** is far behind the ≈34K-parameter MiLTL cascade on the
  decorrelated slices, and it is the model that misses scam-word-free grooming (the bundled
  hard-harm demo cases include its recorded FNs).
- **Cascade ≫ L1** (0.965 vs 0.933 AUROC; F1 0.939 vs 0.840): the Gate-2 SLM arbiter pulls
  the escalate band up (DeLong p < 0.001).
- Residual errors: 21/500 operational decisions (14 FP on cold-benign, 7 FN on low-density
  real phishing) — the adversarial hard core. Journal Youden F1 0.939 vs pooled operational
  F1 0.948 (threshold-convention difference; both computed from the published sheets).

### KorCCViD — transcript-only standard corpus (control)

| Detector | AUROC | F1 | ACC |
|---|---|---|---|
| hf-encoder / tree (frozen) | 1.000 | 1.000 | 1.000 |
| cnn-bilstm (frozen) | 1.000 | 0.997 | 0.998 |
| MiLTL-Cascade | 0.890 | 0.881 | 0.912 |
| Bllossom-B3 | 0.883 | 0.834 | 0.876 |
| lexical (text proxy) | 0.502 | 0.353 | 0.670 |

On their **own corpus** the frozen legacy detectors score a perfect 1.000 — pure corpus
memorization. This is exactly why a KorCCViD-only benchmark is misleading, and why KorMMP
exists. (MiLTL is deliberately *not* tuned to KorCCViD; it runs its transcript-only
compensation path there and still matches the 8B LLM.)

## Published result files

Under [`artifacts/rounds/canonical/`](../artifacts/rounds/canonical/), per seed:

- `consolidated_results.csv` — the journal table above (AUROC/F1/ACC/SEN/SPE/PPV/NPV +
  DeLong significance vs. the MiLTL reference).
- `sheet_{corpus}_{seed}.csv` — per-call records: channels (T/I/F/E/XM), lexical density,
  slice, scenario, score, prediction, outcome, Gate-1 p1, band, decision, transcript
  (where the license allows; see redaction note).
- `sheet_{corpus}_{seed}.l2ledger.jsonl` — the Gate-2 judgment ledger (channel summary,
  prompt, P(harm), XAI rationale) for every escalated call.
- `bundle_{corpus}_{seed}.jsonl` — the exact evaluated bundles.
- `synth_{seed}.jsonl` — the synthetic hard-slice source for that seed.

The 20 bundled demo cases (`demo/cases_canonical.json`) are regenerated from these exact
artifacts by `demo/build_cases.py` — transcripts and per-detector verdicts quoted verbatim.

### Data-licensing redaction (important)

To respect source licenses, transcripts in the **KorMMP** sheets, bundles, and ledgers are
kept inline **only** for self-authored synthetic cases (`synth*`) and FSS/KOGL cases
(`fss*`). Transcripts derived from **AI-Hub** corpora (e.g. `emotion_dialog*`) are redacted:
bundles keep a `transcript_sha256` fingerprint (rebuild per docs/DATA_ACCESS.md and verify
byte-for-byte), and sheet/ledger text is replaced by a marker. All **KorCCViD** files are
kept whole under CC BY-NC-SA 4.0 with attribution (DATA_LICENSE.md). Every reported metric
was computed on the full, unredacted data on the benchmark machine — redaction affects only
the text shipped for inspection, not the numbers.
