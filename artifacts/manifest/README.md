# Benchmark manifests

- `kormmp_real_full.jsonl` — the materialized KorMMP real pool (143 FSS harm + 141 benign
  in the current composition). **FSS rows carry inline transcripts** (KOGL Type-1
  attribution, see ../../DATA_LICENSE.md). **AI-Hub-derived benign rows are pointers**:
  `transcript` is empty and `transcript_sha256` / `transcript_chars` fingerprint the
  original text so a rebuild (docs/DATA_ACCESS.md §2) can be verified byte-exactly.
- `case_inventory.jsonl`, `case_inventory_hard.jsonl`, `case_inventory_callcenter.jsonl` —
  full case inventories (pointer-only: ids, sources, audio path references, stats; no
  transcript content). These document the exact corpus composition of the KorMMP pools.

Regenerate with `scripts/build_case_inventory.py` → `scripts/materialize_kormmp.py` after
obtaining the source corpora (docs/DATA_ACCESS.md).
