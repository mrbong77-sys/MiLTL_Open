# Models — Artifacts and Release Plan

> **Status — released.** The canonical weights are **in this repo now**: the release Calib
> (`calib.release.json`), `channel_extractors.pt`, `miltl_head.pt`, and the canonical
> **Gate-2 LoRA SFT adapter** (`gate2_adapter_1.5b/`, base `Qwen/Qwen2.5-1.5B-Instruct`) —
> all under `artifacts/models/` and auto-detected by the demo. Distribution is split by
> size: small canonical weights live **directly in git**; large frozen-legacy baseline
> weights go to a **GitHub Release**. Every component is also retrainable from this repo
> (docs/REPRODUCIBILITY.md §3).

## Distribution model

Small files live in git (a plain `git clone` gets them); large files (>100 MB, GitHub's
non-LFS per-file limit) are attached to a GitHub Release and fetched by
`scripts/download_models.sh`. Hugging Face Hub is **not** required.

### Committed directly to git (small — a `clone` already has them)

| Artifact | Role | Typical size |
|---|---|---|
| `artifacts/models/channel_extractors.pt` | supplies `Calib` calibration statistics to the analytic channels | a few MB |
| `artifacts/models/calib.release.json` | the release `Calib` as plain JSON (`Calib.to_dict()`) — lets the **CPU demo load it without torch** | a few KB |
| `artifacts/models/miltl_head.pt` | optional TSMixer head (canonical scorer is the analytic rule) | < 1 MB |
| `artifacts/models/gate2_adapter_1.5b/` | Gate-2 LoRA adapter for Qwen/Qwen2.5-1.5B-Instruct (`adapter_config.json` + `adapter_model.safetensors`) | ~30–70 MB |

These paths are white-listed in `.gitignore`. Keep each file **< 100 MB** (GitHub's hard
non-LFS limit); a LoRA adapter is well under that.

**Demo auto-detection.** `demo/miltl_demo.py` picks these up with no flags: the release Calib
(from `calib.release.json`, else extracted from `channel_extractors.pt` when torch is present)
and the Gate-2 SFT adapter (`gate2_adapter_1.5b/`, attached only when the backbone matches the
adapter's `base_model_name_or_path`). Export the JSON calib once from a checkpoint:

```python
import torch, json
from miltl.native.channel_calib import Calib
d = torch.load("artifacts/models/channel_extractors.pt", map_location="cpu")
json.dump(d["calib"], open("artifacts/models/calib.release.json", "w"))
```

### GitHub Release (large — fetched on demand, optional)

| Asset | Role |
|---|---|
| `frozen_korccvid_hf_encoder.tar.gz` | frozen KLUE-RoBERTa-family encoder baseline weights (~400–500 MB) |
| `frozen_korccvid_cnn_bilstm.tar.gz` | frozen CNN-BiLSTM baseline weights |

Publish these as assets on a Release (e.g. tag `models-v1`), then:
`MILTL_RELEASE_TAG=models-v1 bash scripts/download_models.sh` (verifies SHA-256 against
`docs/models.sha256` if present). These are optional — the legacy baselines can be retrained
from `artifacts/frozen/korccvid/train_pool.jsonl` (docs/REPRODUCIBILITY.md §3).

## Shipped in-repo (small, JSON, self-trained)

| File | Role |
|---|---|
| `artifacts/models/wave_seq_fss.json` | Wave-Seq audio-only fairness control (frozen) |
| `artifacts/models/miltl_seq_korccvid.json` | legacy text MiLTL-Seq (used by the MiLTL-Dual naive-fusion control) |
| `artifacts/frozen/korccvid/tree/tree.pkl` | frozen tree-ensemble baseline (pickle — see security note in docs/BASELINES.md) |

## External model dependencies (downloaded from their owners at runtime)

| Model | Used by | License holder |
|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` (and 0.5B ablation) | Gate-2 backbone | Alibaba (Apache-2.0) |
| `Herry443/Llama-8B-KNUT-…` (Bllossom fine-tune) | B3 baseline reproduction | original authors (Llama-3 license terms apply) |
| KLUE-RoBERTa family | `hf_encoder` baseline | KLUE (CC BY-SA) |
