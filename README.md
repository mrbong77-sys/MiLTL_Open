# MiLTL — Multimodal In-call Lightweight Threat Locator

**MiLTL** is an edge-side, real-time Korean voice-phishing (vishing) detector. It observes a
phone call (audio + transcript) in 8-second **nibbles**, encodes each nibble into five
neutrosophic/affect channels **[L,5] = (T, I, F, E, XM)**, and decodes the channel sequence
into a harm decision with a two-stage cascade:

- **Gate-1** (always-on screener, ≈34K-parameter budget, recall-first): the canonical scorer is
  the deterministic, interpretable rule **`risk = E − 2·T + I + XM`** — zero learned parameters
  at inference.
- **Gate-2** (borderline arbiter): a small LLM (Qwen2.5-1.5B + LoRA SFT) that judges only the
  escalated band `p1 ∈ (0.40, 0.90)`, injecting the XM channel summary into its prompt, and
  emits an explainable verdict with recommended edge actions.

The core novelty is **XM — cross-modal contradiction** ("warm text, cold voice"): a
neutrosophic indeterminacy signal that text-only and audio-only detectors are *structurally*
unable to observe. On the lexically-decorrelated hard benchmark **KorMMP** (real FSS phishing
prosody + real-life benign calls, channel-equalized), legacy text, audio-only, and naive
dual-fusion detectors all collapse, while MiLTL survives through XM.

> **Status — complete.** The canonical benchmark (final hard-set run) is done and its result
> sheets are published under [`artifacts/rounds/canonical/`](artifacts/rounds/canonical/) —
> **KorMMP MiLTL-Cascade AUROC 0.965 / F1 0.939**, all baselines included
> ([docs/RESULTS.md](docs/RESULTS.md)).
> **The released weights ship in this repo**: the release Calib
> (`artifacts/models/calib.release.json`), the channel-extractor/head checkpoints, and the
> canonical **Gate-2 LoRA SFT adapter** (`artifacts/models/gate2_adapter_1.5b/`) — the demo
> auto-detects all of them ([docs/MODELS.md](docs/MODELS.md)). Only the large frozen-legacy
> baseline weights are distributed via GitHub Release (and remain retrainable from this repo,
> docs/REPRODUCIBILITY.md). Everything needed to reproduce the benchmark is here; the sole
> exception is AI-Hub-derived text, which licensing forbids redistributing — rebuild it per
> [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md) and verify via the shipped SHA-256 fingerprints.

## Repository map

| Path | Contents |
|---|---|
| `miltl/native/` | Channel-bottleneck core: `channel_calib` (canonical channels), `nibble_features` (featurizer), `seq_head` (optional TSMixer head), `explain` (XAI) |
| `miltl/nibble/` | Nibble/prosody core: segmentation, prosody, audio decode (telephone band + μ-law equalization), corpora adapters |
| `miltl/baseline/` | Benchmark harness (pure stdlib): metrics, result sheets, ASR sidecar resolution |
| `adapters/baselines/` | Detectors: **`native_channel` (MiLTL Gate-1/cascade)** · `gate2_slm` (Gate-2 SLM) · `wave_seq` (audio-only + naive-fusion fairness controls) · `hf_encoder` / `cnn_bilstm_fasttext` / `tree_ensemble` / `bllossom_repro` (legacy baselines) |
| `scripts/` | Canonical pipeline: `canonical_bench.py` (single orchestrator), synthesis/composition/materialization, Gate-2 SFT, provenance audit, data collectors |
| `demo/` | **Standalone CPU demo (web UI, no GPU)** — live Gate-1/Gate-2 cascade with channel traces, latencies, CPU/RAM meters; bundled canonical cases carrying real per-detector verdicts; paste-text and wave-upload (real prosody + auto-ASR) inputs; edge budget benchmark ([demo/README.md](demo/README.md)) |
| `configs/` | Operating points and calibration conventions (`nibble.yaml`, `targets.yaml`, `data.yaml`, `datasets.yaml`) |
| `artifacts/` | Benchmark manifests, synthetic seed bundles, frozen KorCCViD pools, self-trained fairness-control weights (JSON), FSS board summaries |
| `data/raw/fss/` | FSS "The Scammer's Voice" text corpus (post metadata + dialogue scripts; audio is fetched from the FSS site — see docs/DATA_ACCESS.md) |
| `docs/` | English documentation set (below) |
| `tests/` | GPU-free regression tests: `python -m unittest discover -s tests` |

## Documentation

| Doc | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Inference path, channel definitions, calibration, cascade banding, XAI |
| [docs/BENCHMARK.md](docs/BENCHMARK.md) | KorMMP design: lexical decorrelation, hard slices, fairness controls, freeze protocol, result-sheet schema |
| [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md) | How every corpus was obtained (FSS, AI-Hub, KorCCViD) and how to re-obtain it from the original sources |
| [docs/BASELINES.md](docs/BASELINES.md) | Baseline provenance and reproduction (Bllossom-B3, Wave-Seq, frozen legacy detectors, Gate-2 backbone) |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | End-to-end reproduction: environment → data → training → canonical benchmark |
| [docs/MODELS.md](docs/MODELS.md) | Model artifacts and the Hugging Face Hub release plan |
| [docs/RESULTS.md](docs/RESULTS.md) | Canonical result sheets (published after the final benchmark run) |
| [DATA_LICENSE.md](DATA_LICENSE.md) | Per-source data licensing and redistribution policy |

## Quick start

```bash
pip install -r requirements.txt          # core (numpy, pyyaml, FSS collector deps)
python -m unittest discover -s tests     # GPU-free regression suite

# Generate the synthetic hard-case corpus deterministically (matches the committed bundles):
python scripts/synth_edgecases.py --n-per 40 --seed 42 --out /tmp/synth_check.jsonl

# Try the cascade interactively on a CPU-only laptop (web UI at :7861):
pip install -r demo/requirements-demo.txt
python demo/miltl_demo.py          # release Calib auto-loads; see demo/README.md for details

# Full canonical Gate-2 (1.5B + released SFT adapter, auto-attached):
pip install torch transformers peft
python demo/miltl_demo.py --gate2-model Qwen/Qwen2.5-1.5B-Instruct
```

Full benchmark reproduction (needs GPU + original audio corpora) is described step-by-step in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Data policy (important)

- **Synthetic sources are shipped verbatim** (`scripts/synth_edgecases.py` scenario banks +
  `artifacts/rounds/canonical/synth_*.jsonl`) — full transparency for the hard-benign /
  hard-harm slices.
- **FSS-derived text** (public Financial Supervisory Service bulletin content) is included
  with source attribution (KOGL Type 1).
- **AI-Hub-derived transcripts are NOT redistributed** (license restriction). The benchmark
  manifests keep those rows as *pointers* (case id, source, SHA-256 of the transcript);
  [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md) documents exactly which AI-Hub datasets were
  used and how to rebuild the identical pool with the shipped scripts.
- **KorCCViD** pools are included under CC BY-NC-SA 4.0 with attribution.
- No raw audio of any kind is stored in this repository.

## License

Code: [Apache-2.0](LICENSE). Data: per-source terms in [DATA_LICENSE.md](DATA_LICENSE.md).
