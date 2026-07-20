# Data Access — Sources, Acquisition Paths, Rebuild Guide

Every corpus used by KorMMP and the training pipeline, where it came from, and how a
researcher or institution can re-obtain it from the original source. Redistribution policy
per source is in [DATA_LICENSE.md](../DATA_LICENSE.md).

## 0. Summary table

| Source key (manifests) | Corpus | Provider | Role | In this repo |
|---|---|---|---|---|
| `fss`, `fss_audio` | "The Scammer's Voice" real vishing recordings + dialogue scripts | FSS (Financial Supervisory Service, Korea) | **TEST-only harm** | text inline (`data/raw/fss/`), audio: fetch yourself |
| `korccvi` | KorCCViD v1.3 transcripts | Boussougou & Park (academic, GitHub) | frozen-legacy training + transcript-only control bench | pools inline (`artifacts/frozen/korccvid/`) |
| `ksponspeech_dual` | KsponSpeech (Korean speech, pcm + trn) | AI-Hub | dual-modal benign | pointer only |
| `emotion_dialog` | Emotion-tagged free conversation (wav + json) | AI-Hub | dual-modal benign | pointer only |
| `sample_voice` | Sample_voice (wav + json) | AI-Hub | dual-modal benign | pointer only |
| `freetalk` | Free conversation speech — general public (STT text) | AI-Hub | text benign | pointer only |
| `aihub` | Multi-session / purpose / topic / daily dialogue JSON corpora (datasets 011, 020, 021, 141, 297, …) | AI-Hub | text benign | pointer only |
| `aihub_tsv` | Empathetic dialogue 046 (+ persona 044) TSV | AI-Hub | text benign | pointer only |
| `callcenter_ktel`, `callcenter_minwon` | call-center consultation + civil-complaint dialogs (e.g. dataset 143) | AI-Hub | **hard-negative benign** (finance/authority vocabulary) | pointer only |
| `synth_hh_*`, `synth_hb_*` | synthetic hard-harm / hard-benign | this project | hard slices + Gate-2 SFT | **fully inline** |

Corpus adapters that parse each layout: `miltl/nibble/corpora.py` (one function per source,
documented per-schema). Inventory counts per source are recorded in
`artifacts/manifest/case_inventory*.jsonl`, and the exact AI-Hub folder inventory used
(file counts, extensions, audio durations) is preserved in
`artifacts/normal_corpus/structure.md`.

## 1. FSS — "The Scammer's Voice" (그놈 목소리)

Real voice-phishing call recordings and dialogue scripts published by the Korean Financial
Supervisory Service for public awareness, on the *Voice Phishing Keeper*
(보이스피싱지킴이) boards at `https://www.fss.or.kr`:

- Voice board: `B0000206` (`menuNo=200690`) — posts with scammer call audio (mp4/mp3 via a
  JS player) + dialogue script text.
- Voice board 2: `B0000207` (`menuNo=200691`) — 227 posts / 227 mp3 files (summary:
  `artifacts/fss_sources/voice2.json`).
- Video board — video-format releases (summary: `artifacts/fss_sources/video.json`).

**What we used**: 122 text-script posts (`fss`) + 506 audio cases (`fss_audio`) as the
TEST-only harm pool; 143 of the audio cases are materialized into the current bench pool
(`scripts/materialize_kormmp.py --harm 506` extends to the full pool).

**How to re-obtain** (run from a Korean-network machine; the boards block some overseas /
datacenter ranges):

```bash
pip install -r requirements.txt
# discover without downloading:
python scripts/fetch_fss_voicephishing.py --pages 1-5 --dry-run
# collect audio + post text + metadata:
python scripts/fetch_fss_voicephishing.py --pages 1-20 --out data/raw/fss --resume
```

The collector handles the eGovFrame BBS layout, JS-injected media, and `getFile.do`
attachment-id reconstruction; `--analyze-html` and `--selftest` support offline debugging
if the site layout changes. Post text/metadata land in `data/raw/fss/` — the same layout as
the copies shipped in this repo, so you can verify your crawl against ours (`index.jsonl`
per board, `posts/<nttId>/meta.json`).

Transcripts of the audio are generated locally with `scripts/asr_transcribe.py`
(faster-whisper; `pip install -r requirements-asr.txt`), producing `*.asr_light.txt`
sidecars next to each audio file — the layout `scripts/materialize_kormmp.py` expects.

## 2. AI-Hub benign corpora

[AI-Hub](https://aihub.or.kr) is Korea's public AI-data portal (NIA). Datasets are free but
require a (Korean) account and per-dataset terms agreement, and **may not be redistributed**
— which is why every AI-Hub-derived transcript in this repo is a pointer, not content.

Datasets used (search AI-Hub by the Korean titles; numeric prefixes are AI-Hub catalog numbers):

| AI-Hub dataset (Korean title) | Used as |
|---|---|
| 011. 일상대화 한국어 멀티세션 데이터 | `aihub` text benign |
| 020. 주제별 텍스트 일상 대화 데이터 | `aihub` text benign |
| 021. 용도별 목적대화 데이터 | `aihub` text benign |
| 141. 한국어 멀티세션 대화 | `aihub` text benign |
| 297. SNS 데이터 고도화 | `aihub` text benign |
| 044. 페르소나 대화 / 046. 공감형 대화 | `aihub_tsv` text benign |
| 자유대화 음성(일반남녀) | `freetalk` text benign (per-utterance STT) |
| 한국어 음성 (KsponSpeech, 평가용) | `ksponspeech_dual` dual-modal benign (pcm + trn) |
| 감정이 태깅된 자유대화 | `emotion_dialog` dual-modal benign |
| Sample_voice | `sample_voice` dual-modal benign |
| 143. 민원 업무 효율·자동화 언어 AI 학습데이터 (+ call-center consultation corpus) | `callcenter_*` hard-negative benign |

**How to rebuild the benign pool**:

```bash
# 1) download the datasets from AI-Hub into data/raw/normal/ keeping original folder names
#    (zip archives can stay zipped — adapters read zip members directly)
# 2) rebuild the case inventory (audio probing needs ffmpeg):
python scripts/build_case_inventory.py --out artifacts/manifest/case_inventory_hard.jsonl
# 3) generate ASR sidecars where needed (GPU):
python scripts/asr_transcribe.py ...
# 4) materialize the bench pool; verify against the shipped pointer manifest:
python scripts/materialize_kormmp.py --inventory artifacts/manifest/case_inventory_hard.jsonl \
    --harm 506 --benign 600 --out artifacts/manifest/kormmp_real_full.jsonl
```

Each pointer row in the shipped `artifacts/manifest/kormmp_real_full.jsonl` carries
`transcript_sha256` and `transcript_chars` so you can verify byte-exact reconstruction of
the benign transcripts you rebuild.

## 3. KorCCViD (transcript corpus)

KorCCViD v1.3 — Korean Call Content Vishing transcripts (real vishing call transcriptions +
benign call texts), from M. K. M. Boussougou & D.-J. Park's public research repository:
`https://github.com/selfcontrol7/Korean_Voice_Phishing_Detection` (dataset license
CC BY-NC-SA 4.0). Used to (a) train/freeze the legacy baselines and MiLTL calibration, and
(b) serve as the transcript-only standard bench (control corpus). The frozen evaluation and
training pools are shipped at `artifacts/frozen/korccvid/{test_pool,train_pool}.jsonl`;
`scripts/materialize_korccvid.py` rebuilds them from the upstream CSV.

## 4. Synthetic sources (shipped verbatim)

The synthetic hard-harm / hard-benign generator with its full Korean scenario banks is
`scripts/synth_edgecases.py`; generation is deterministic given `--seed`, and the exact
bundles consumed by the canonical bench are committed at
`artifacts/rounds/canonical/synth_*.jsonl`. No external data is needed to reproduce them:

```bash
python scripts/synth_edgecases.py --n-per 200 --seed <seed> --out synth_<seed>.jsonl
```
