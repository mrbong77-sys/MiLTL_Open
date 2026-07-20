# Reproducibility — End-to-End Guide

Two tiers: **GPU-free** (verify pipeline logic, result consolidation, synthetic corpus,
provenance audit math) and **full** (rebuild data pools, train, run the canonical bench).

## 0. Environment

```bash
python >= 3.10
pip install -r requirements.txt                      # core
pip install -r adapters/baselines/requirements.txt   # baselines + Gate-2 (GPU)
pip install -r requirements-asr.txt                  # ASR sidecar generation (GPU)
# ffmpeg is required for audio decode (mp4/mp3 → PCM)
```

## 1. GPU-free verification (works on any machine, no external data)

```bash
python -m unittest discover -s tests        # regression suite (banding, ledger, consolidation, audio fairness)

# Deterministic synthetic corpus — regenerate and compare to the committed bundles:
python scripts/synth_edgecases.py --n-per 40 --seed 42 --out /tmp/synth42.jsonl

# Consolidate a journal table from existing sheets (once results are present):
python scripts/consolidate_results.py --dir artifacts/rounds/canonical
```

## 2. Rebuild the data pools (see docs/DATA_ACCESS.md for acquisition)

```bash
# FSS harm pool (Korean network):
python scripts/fetch_fss_voicephishing.py --pages 1-20 --out data/raw/fss --resume
# AI-Hub benign corpora → data/raw/normal/ (manual download, keep folder names)

# ASR sidecars for all audio (GPU):
python scripts/asr_transcribe.py ...

# Case inventory + materialized bench pool:
python scripts/build_case_inventory.py --out artifacts/manifest/case_inventory_hard.jsonl
python scripts/materialize_kormmp.py --inventory artifacts/manifest/case_inventory_hard.jsonl \
    --harm 506 --benign 600 --out artifacts/manifest/kormmp_real_full.jsonl
# Verify benign transcripts against the shipped pointer manifest (transcript_sha256 match).

# KorCCViD pools (or use the shipped ones):
python scripts/materialize_korccvid.py ...
```

## 3. Models

Either download the released weights (`bash scripts/download_models.sh`, see
docs/MODELS.md) or retrain:

```bash
# channel extractor checkpoint (supplies Calib statistics):
python scripts/train_channel_extractors.py ...
# optional TSMixer head (canonical scorer is analytic; head only used with blend<1):
python scripts/train_miltl_head.py ...

# Gate-2 LoRA SFT (frozen protocol: KorCCViD-train + seed-99 synthetic, fair audio):
python scripts/synth_edgecases.py --n-per 200 --seed 99 --out artifacts/rounds/synth_99.jsonl
python scripts/compose_hard_kormmp.py --seed 99 --synth artifacts/rounds/synth_99.jsonl \
    --total 300 --fair-audio --out artifacts/rounds/train_fair_99.jsonl
python scripts/gate2_sft.py --train-bundle artifacts/rounds/train_fair_99.jsonl --band all \
    --fair-audio --gate2-model Qwen/Qwen2.5-1.5B-Instruct \
    --ckpt artifacts/models/channel_extractors.pt --head artifacts/models/miltl_head.pt \
    --out artifacts/models/gate2_adapter_1.5b --epochs 3 --device cuda
```

## 4. Canonical benchmark (single orchestrator)

```bash
python scripts/canonical_bench.py --rounds 5 --total 100 --device cuda \
    --ckpt artifacts/models/channel_extractors.pt --head artifacts/models/miltl_head.pt \
    --gate2-adapter artifacts/models/gate2_adapter_1.5b

# Exact-seed reproduction of a published run: pass the recorded seeds, e.g.
python scripts/canonical_bench.py --seeds <seed1,seed2,...> ...
```

Outputs under `artifacts/rounds/canonical/`: `seeds.json`, `consolidated_results.csv`
(journal table), `sheet_{corpus}_{seed}.csv` (per-call), `sheet_*.l2ledger.jsonl` (Gate-2
judgment ledger), `bundle_*.jsonl` / `synth_*.jsonl` (round bundles).

Post-run checks:

```bash
python scripts/audit_audio_provenance.py ...   # source-shortcut probe must stay low
python scripts/lexical_shortcut.py ...         # decorrelation probe must fail on hard slices
```

## 5. What can be reproduced without audio access

The inline portions of this repo (FSS text, KorCCViD pools, synthetic bundles) support the
transcript-path benchmarks and all consolidation/audit logic without any external download.
Prosody/XM reproduction requires the original audio (FSS crawl + AI-Hub download) — this is
a licensing constraint of the source corpora, not a technical one (see DATA_LICENSE.md).
