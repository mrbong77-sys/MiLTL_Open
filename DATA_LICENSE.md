# Data Licensing and Redistribution Policy

The **code** in this repository is Apache-2.0 ([LICENSE](LICENSE)). The **data** files are
governed per source as follows.

## 1. Synthetic corpus (this project) — included, no restriction

`scripts/synth_edgecases.py` scenario/filler banks and every `synth_*` bundle
(`artifacts/rounds/canonical/synth_*.jsonl`) are original works of this project, released
under the repository's Apache-2.0 terms. They contain no material from FSS, AI-Hub, or
KorCCViD.

## 2. FSS "The Scammer's Voice" — text included with attribution; audio not redistributed

`data/raw/fss/` (post metadata + dialogue script text) derives from the Financial
Supervisory Service's public voice-phishing awareness boards (보이스피싱지킴이, fss.or.kr).
FSS website content is provided under Korea Open Government License (KOGL) **Type 1
(attribution)** unless marked otherwise. Attribution: **Source — Financial Supervisory
Service (금융감독원), Voice Phishing Keeper boards.** We do not redistribute the audio
recordings; obtain them from the FSS boards directly (docs/DATA_ACCESS.md §1). If FSS's
terms for specific posts differ, the FSS terms prevail — report any concern via an issue
and the affected content will be removed.

## 3. AI-Hub corpora — NOT redistributed (pointers only)

AI-Hub dataset terms prohibit redistribution of the data (including derived transcripts).
Therefore this repository contains **no AI-Hub content**: benchmark rows derived from
AI-Hub corpora appear only as pointers (case id, source key, `transcript_sha256`,
`transcript_chars`, audio path reference). Researchers rebuild these rows from AI-Hub
under their own account/terms (docs/DATA_ACCESS.md §2) and can verify byte-exact
reconstruction against the SHA-256 fingerprints.

## 4. KorCCViD — included under CC BY-NC-SA 4.0

`artifacts/frozen/korccvid/{test_pool,train_pool}.jsonl` derive from **KorCCViD v1.3**
(M. K. M. Boussougou & D.-J. Park, `github.com/selfcontrol7/Korean_Voice_Phishing_Detection`),
licensed **CC BY-NC-SA 4.0**. Accordingly, these two files (and any derivative of them) are
redistributed under **CC BY-NC-SA 4.0 with attribution to the original authors** — they are
*not* Apache-2.0, and commercial use of these files is not permitted. Please cite the
original authors' work when using them.

## 5. Model weights

External backbones (Qwen2.5, Bllossom/Llama-8B fine-tune, KLUE-RoBERTa) remain under their
owners' licenses and are downloaded from their official sources at runtime. Weights trained
by this project are released via Hugging Face Hub under terms stated in the release
(docs/MODELS.md).

## 6. Privacy note

FSS recordings are published by a government agency for public awareness and are handled
here in text form only. If you believe any shipped text contains personal information,
open an issue; it will be removed promptly.
