#!/usr/bin/env bash
# Resolve the MiLTL model artifacts needed to run the canonical pipeline.
#
# Distribution model (see docs/MODELS.md):
#   * SMALL canonical backbone — channel_extractors.pt, miltl_head.pt, and the Gate-2 LoRA
#     adapter (artifacts/models/gate2_adapter_1.5b/) — are committed DIRECTLY to this repo,
#     so a plain `git clone` already has them. This script just verifies they are present.
#   * LARGE frozen-legacy weights (full hf-encoder / cnn-bilstm) exceed GitHub's 100 MB
#     per-file limit, so they are attached to a GitHub Release and fetched on demand below.
#     They are OPTIONAL — every baseline can be retrained from artifacts/frozen/korccvid/
#     train_pool.jsonl (docs/REPRODUCIBILITY.md §3).
set -euo pipefail

CORE=(
  "artifacts/models/channel_extractors.pt"
  "artifacts/models/miltl_head.pt"
  "artifacts/models/gate2_adapter_1.5b/adapter_config.json"
)
echo "[download_models] checking in-repo canonical backbone…"
missing=0
for f in "${CORE[@]}"; do
  if [[ -f "$f" ]]; then echo "  ok  $f"; else echo "  MISSING  $f"; missing=1; fi
done
if [[ "$missing" == 1 ]]; then
  echo "[download_models] core weights are committed to git — if missing, your checkout is"
  echo "  incomplete, or they have not been pushed from the training machine yet"
  echo "  (docs/MODELS.md), or retrain them per docs/REPRODUCIBILITY.md §3."
fi

# --- optional: large frozen-legacy weights from a GitHub Release ---
# Fill RELEASE_TAG once the assets are published; until then this section is a no-op.
RELEASE_TAG="${MILTL_RELEASE_TAG:-}"          # e.g. models-v1
REPO="${MILTL_REPO:-mrbong77-sys/MiLTL_Open}"
LEGACY_ASSETS=(
  "frozen_korccvid_hf_encoder.tar.gz"
  "frozen_korccvid_cnn_bilstm.tar.gz"
)
if [[ -z "$RELEASE_TAG" ]]; then
  echo "[download_models] (frozen-legacy weights: set MILTL_RELEASE_TAG to fetch from a Release;"
  echo "                   optional — baselines are retrainable from the shipped train_pool.jsonl)"
  exit 0
fi
echo "[download_models] fetching frozen-legacy weights from Release $RELEASE_TAG…"
mkdir -p artifacts/frozen/korccvid
for a in "${LEGACY_ASSETS[@]}"; do
  url="https://github.com/${REPO}/releases/download/${RELEASE_TAG}/${a}"
  echo "  GET $url"
  curl -fsSL "$url" -o "artifacts/frozen/korccvid/$a"
  tar -xzf "artifacts/frozen/korccvid/$a" -C artifacts/frozen/korccvid/
done
if [[ -f docs/models.sha256 ]]; then
  echo "[download_models] verifying checksums…"; sha256sum -c docs/models.sha256
fi
echo "[download_models] done."
