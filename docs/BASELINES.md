# Baselines — Provenance and Reproduction

All baselines are evaluated by the same harness (`miltl/baseline/`), on the same bundles,
with the same metrics and result-sheet format as MiLTL. Heavy dependencies are isolated in
`adapters/baselines/requirements.txt`.

## 1. Bllossom-B3 (Korean vishing LLM — original-method reproduction)

- **Origin**: the *VP_detector_SLM* line of work (public repo `kufany/VP_detector_SLM`;
  method also described in the KorCCViD authors' publications) — a Korean Llama-8B
  (Bllossom) fine-tuned for vishing likelihood scoring.
- **Weights**: the publicly released fine-tuned full model on Hugging Face Hub
  (`Herry443/Llama-8B-KNUT-…` — LoRA merged and pushed by the original authors). We load it
  4-bit for inference memory only; weights are frozen, so quantization does not affect
  reproduction claims.
- **Method faithfully reproduced** (`adapters/baselines/bllossom_repro.py`): the original
  system prompt + prompt format + 11-criteria checklist (Korean, kept byte-identical), CoT
  generation, and the "따라서 가능도는 [N]" parse → N ∈ 0..10 → score = N/10 (continuous, so
  AUROC is threshold-free). Point-metric thresholds are selected on train only (or γ_th as
  in the original notebooks).
- This adapter is a **reproduction**, not an improvement. `bllossom_llm.py` is our separate
  single-token reimplementation (kept for the checklist constants it shares).

## 2. Wave-Seq (audio-only fairness control)

- **What**: prosody-sequence CNN over telephone-band prosody vectors
  (`miltl/nibble/wave_seq.py` + `adapters/baselines/wave_seq.py`), trained on FSS audio.
- **Weights shipped**: `artifacts/models/wave_seq_fss.json` (self-trained, JSON, frozen).
- **Why it exists**: fairness self-evidence. It answers "would audio access alone win?" —
  after source equalization it collapses to chance on KorMMP, proving MiLTL's margin is not
  modality access. Its earlier high score *before* equalization was shown by
  `scripts/audit_audio_provenance.py` to be a corpus-channel shortcut.
- **MiLTL-Dual** (same module): naive text ⊕ wave normalized max-OR fusion, no XM — the
  naive-fusion control. Uses `artifacts/models/miltl_seq_korccvid.json` (shipped) for the
  text side.

## 3. Frozen legacy text detectors (KorCCViD-trained)

| Adapter | Model | Frozen artifacts |
|---|---|---|
| `hf_encoder.py` | Korean transformer encoder classifier (KLUE-RoBERTa family) | `artifacts/frozen/korccvid/hf_encoder/` (config + tokenizer shipped; weights in the model release) |
| `cnn_bilstm_fasttext.py` | CNN-BiLSTM + FastText (per the original Korean vishing paper) | `artifacts/frozen/korccvid/cnn_bilstm/vocab.json` (+ weights in the model release) |
| `tree_ensemble.py` | feature-tree ensemble (CatBoost/LGBM) | `artifacts/frozen/korccvid/tree/tree.pkl` |

All three are trained once on KorCCViD-train (`train_pool.jsonl`, shipped) and frozen;
`artifacts/frozen/korccvid/meta.json` records the freeze. On KorCCViD-test they score
near-perfectly (corpus memorization — the motivating observation for KorMMP); the KorMMP
hard slices are where they collapse. Retraining from the shipped pools reproduces the
frozen models; released weight files (docs/MODELS.md) reproduce them exactly.

> Security note: `tree.pkl` is a Python pickle. Load it only in an isolated environment, or
> retrain from `train_pool.jsonl` if you prefer not to unpickle third-party files.

## 4. MiLTL Gate-2 backbone

- **Canonical**: `Qwen/Qwen2.5-1.5B-Instruct` + LoRA SFT (`scripts/gate2_sft.py`), trained
  only on KorCCViD-train + the seed-99 synthetic bundle (never KorMMP), with `--fair-audio`
  to match train/eval audio conditioning. The 0.5B backbone is an edge-lightweight ablation.
- The LoRA adapter is part of the model release (docs/MODELS.md); the SFT script and its
  exact data path are shipped so it can be retrained end-to-end.

## 5. Lexical proxy

`lexical(text-proxy)` (`scripts/lexical_shortcut.py` / harness-internal): scam-keyword
density scorer. It is the decorrelation probe — by construction it must fail on hard
slices; if it doesn't, the slice composition is broken.
