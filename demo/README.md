# MiLTL Standalone Demo — CPU-only, Web UI

> ⚠️ **Korean-only / 한국어 전용.** MiLTL is a Korean vishing detector: the channel
> lexicons, calibration, Gate-2 prompts, and the auto-ASR (`language=ko`) are all
> Korean-based. English or other-language transcripts/audio will **not** be judged
> correctly — use Korean input.

Experience the full MiLTL cascade **on an ordinary laptop, with no GPU**: live per-nibble
channel traces (T/I/F/E/XM), the Gate-1 risk trajectory and banding, Gate-2 SLM activation
with its P(harm) verdict and XAI rationale, and real-time stage latencies plus CPU/RAM
occupancy of the demo process. Its purpose is the same as the PEINN routing demo's:
make the edge-deployability claim *tangible and measurable*, not just stated.

```bash
# from the repo root
pip install -r demo/requirements-demo.txt      # numpy only for L1
python demo/miltl_demo.py                      # web UI at http://localhost:7861
python demo/miltl_demo.py --selftest           # terminal sanity check (no UI)

# enable the Gate-2 (L2) panel — CPU-only torch (a CUDA/MPS GPU is auto-detected & used if present):
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install transformers
python demo/miltl_demo.py --gate2-model Qwen/Qwen2.5-0.5B-Instruct   # default (light; zero-shot)
pip install peft                                                      # for the released SFT adapter
python demo/miltl_demo.py --gate2-model Qwen/Qwen2.5-1.5B-Instruct   # canonical: 1.5B + SFT adapter
# With the 1.5B backbone the released LoRA SFT adapter (artifacts/models/gate2_adapter_1.5b/)
# is attached automatically — that is the exact canonical Gate-2 from the benchmark.
# The panel shows the detected runtime and an "Enable Gate-2" toggle (auto-on with a GPU,
# off on CPU-only, since a 0.5B backbone on CPU tends to flip benign→harm — trust Gate-1 there).

# enable wave-only auto-transcription (ASR):
pip install faster-whisper
# GPU ASR (e.g. DGX): faster-whisper's CTranslate2 backend is a SEPARATE build from torch.
# A CPU-only CTranslate2 wheel raises "not compiled with CUDA support" even on a GPU box — the
# demo then auto-falls back to CPU int8 (and says so). For true CUDA fp16 ASR, install a
# CUDA-enabled CTranslate2 matching your CUDA/cuDNN (see the CTranslate2 install docs).
```

## Three ways to feed it

1. **Bundled canonical cases (20 scenarios)** — transcripts quoted **verbatim from the
   5-seed canonical benchmark**, each carrying the **real per-detector verdicts recorded in
   that run**. This is how the demo shows *why legacy detectors fail and MiLTL succeeds* —
   from actual results, not a scripted claim. (These replaced the earlier hand-built
   samples, which carried FSS post captions like "수사기관 사칭형(검찰, 경찰)" / dates /
   "사기범:" that leaked the answer.) The pool spans the failure modes so each legacy
   detector *type* is legible — not every case is a legacy failure:
   - **Legacy home turf (KorCCViD)** — prosecutor / police impersonation: *every* legacy
     detector is correct (in-corpus memorization → why a KorCCViD-only score misleads).
   - **Easy-harm, keyword-dense (real FSS)** — even the pure lexical keyword matcher fires
     correctly: the lexical proxy is *just* a keyword matcher, fine when scam words are
     present, blind when they are not.
   - **Hard-benign ×8** (bank / card / insurance / telecom / delivery / gov office /
     hospital / securities) — legitimate finance calls where hf-encoder / cnn-bilstm / tree
     / lexical false-positive ("finance words = scam"); MiLTL stays benign.
   - **Hard-harm ×7** (family / institution / loan / refund / investment / subsidy /
     delivery impersonation) — scam-word-free grooming the 8B LLM and audio-only Wave-Seq
     miss (FN); MiLTL catches it on XM + prosody.
   - **Real FSS recordings** — raw ASR, including a case that exposes the weak CatBoost tree
     (misses harm the hf-encoder catches).
2. **Paste any Korean transcript** — text-only path (Gate-1 runs its lexical-F compensation
   since there is no audio).
3. **Upload a `.wav` / `.mp3` / `.mp4` / `.m4a`** — **real prosody is extracted** (compressed
   formats decode via the pip-bundled `imageio-ffmpeg`, so no system ffmpeg install is needed)
   from your audio. If you leave the transcript blank, it is **auto-transcribed** (faster-whisper;
   CUDA fp16 if a GPU is present, else CPU int8) and the transcript is **written back into the
   INPUT box** (with a "🎙 auto-transcribed" badge) so you can see exactly the text the lexical
   keyword-density proxy and the CatBoost tree ensemble are judging. With an uploaded file the
   player exposes a **Live streaming** toggle (on by default when ASR is installed): see below.
   This is the genuine multimodal path, not simulation.

## Live streaming — true online, honest about the hardware

Turn on **Live streaming** and the demo plays the call while processing it **online, one 8 s nibble
at a time, no look-ahead**. The call arrives in **real time (1×)** like a live phone call — nibble
*k*'s audio exists only at real-time *k*×8 s, so the pipeline waits for it and **never runs ahead of
playback**. The edge question is *does each nibble finish inside its 8 s budget?*, not *how fast can
it batch a file?*. The panel shows:

- **per-nibble compute / keeping up?** — the latest nibble's compute vs the **8 s budget**
  (green = keeps up, red = falls behind). The edge signal.
- **RTF = compute ÷ audio** (excludes the real-time wait), for the **ASR front-end** vs **MiLTL's
  cascade** separately. ASR dominates; MiLTL stays far under 1× on any device.

**The verdict is confirmed mid-stream.** As soon as the observation anchor (80 words) is met — not
at the end of the window — the Gate-1 band locks and shows **SAFE / HARM / ESCALATE→L2**; on
escalate you can score Gate-2 right away. A 3-minute-plus call therefore decides early and does not
wait for the full ~208 s deadline below.

**Decision deadline (bounded observation).** MiLTL is an *in-call early-warning* detector: it
does not wait for a 37-minute call to end. The verdict is locked within the **canonical
observation envelope — `MAX_NIBBLES` = 26 nibbles × 8 s = 208 s (~3.5 min)** — the same cap the
benchmark `featurize_channels` enforces on scoring. A longer upload keeps playing in the browser,
but ASR + scoring **stop at the deadline** and the verdict is fixed there (the panel shows
`⏱ decision deadline` and, for a long call, a note that the remainder is not scored). This also
stops the runaway transcription a long file used to cause. Override with `--max-nibbles N`
(clamped to `MAX_NIBBLES`, since scoring cannot see past the envelope); a smaller value simulates
an *earlier* forced decision.

**This is deliberately hardware-dependent — that dependence *is* the point.** The streaming
buffer is dominated by the **ASR front-end** (here a faster-whisper stand-in: CPU int8, or CUDA
fp16 when a GPU is present). On a CPU-only notebook that stand-in can exceed real-time and the
buffer goes negative; on a GPU — or a modern phone's **on-device ASR/NPU**, which is what a real
deployment uses, not our whisper — it runs well under `1×`. **MiLTL's own cascade (always-on
Gate-1 + a rare single-forward Gate-2) stays far under `1×` on every device** (typically a few
percent of real-time), which is the edge-deployability claim the whole project rests on. The
MiLTL RTF shown is a *conservative upper bound*: the demo re-scans the whole prefix each nibble
for simplicity, whereas an on-device build keeps incremental channel state and only featurizes
the newest nibble.

For bundled/pasted inputs without audio you can also attach a **simulated prosody profile**
(warm / cold / cold-pressure) so the XM cross-modal mechanics are observable. Stream the
call nibble-by-nibble, and when Gate-1 lands in the escalate band, trigger the Gate-2 SLM
score (a single CPU forward pass) and an optional generated rationale.

## What runs where

| Stage | Implementation | Needs |
|---|---|---|
| L1 Gate-1 | `featurize_channels` → calibrated channels `[L,5]` → analytic rule `risk = E − 2·T + I + XM` → banding τ 0.40/0.90 | numpy only |
| Real audio | uploaded wav/mp3/mp4/m4a → `decode_to_pcm` → real `prosody_stream` (telephone-band + μ-law equalized) → channels | numpy + imageio-ffmpeg (pip-bundled ffmpeg; no system install) |
| Auto-ASR | wave-only upload → faster-whisper (CUDA fp16 if a GPU is present, else CPU int8) → transcript. In **Live streaming** mode: online per-8s-nibble, no look-ahead | faster-whisper (optional) |
| L2 Gate-2 | the repo's real `Gate2SLM` (prompt, yes/no token log-prob scoring, rationale generation). The runtime is **auto-detected**: CUDA/Apple-MPS GPU → fp16, else CPU → fp32. The panel exposes an **Enable Gate-2** toggle (auto-on when a GPU is present, off on CPU-only). | torch + transformers (optional) |
| Meters | per-stage wall latency; process CPU% and RSS from `/proc` | stdlib |

## Budget review — is L2 really laptop/mobile-feasible?

Measured with `demo/edge_benchmark.py` on this project's CI container
(**4 vCPU, 16 GB RAM, fp32, no GPU** — re-run on your own machine; latency belongs to the
silicon it was measured on):

| Stage | Params | Process RAM | Latency (measured) |
|---|---|---|---|
| **L1 Gate-1** (always-on) | **0 learned** (Calib = 42 scalars) | **~45 MB** | **0.35 ms**/full call (median), 0.02 ms/nibble, ~920 calls/s |
| **L2** Qwen2.5-**0.5B** fp32 | 0.49 B | ~2.5 GB | **score 2.6–3.3 s** (1 forward, ~800-token prompt) · rationale ~11 tok/s · load 2–20 s |
| **L2** Qwen2.5-**1.5B** fp32 (canonical) | 1.54 B | ~9.6 GB peak | **score 7.3 s** · rationale 4.8 tok/s · load 32 s |

Why this supports the mobile claim:

- **L1 is effectively free.** Sub-millisecond per call against an 8-second observation
  cadence is a ~0.005 % duty cycle; the always-on screener fits any phone-class CPU with
  tens of MB of memory and zero learned weights.
- **L2 fires rarely and needs no generation to decide.** Only escalate-band calls
  (`0.40 < p1 < 0.90`) reach Gate-2, and its *decision* is one forward pass over the
  prompt (yes/no token log-probability families) — seconds, once per suspicious call, not
  a streaming cost. The generated rationale is an optional XAI extra.
- **fp32 is the worst case.** The numbers above are the most conservative CPU path.
  An int4-quantized 1.5B runtime is ≈1 GiB of weights (0.5B ≈ 0.3 GiB) and typically
  several times faster on the same cores; phone-class NPUs/CPUs run 1.5B-class models at
  interactive rates today. A 16 GB laptop runs the canonical 1.5B even in fp32.
- Single-core emulation: `python demo/edge_benchmark.py --gate2 --threads 1`.

## Why legacy detectors fail and MiLTL succeeds — from the actual benchmark

Selecting a **bundled case** shows the *recorded* verdicts of the **full detector panel**
from the 5-seed canonical run (`artifacts/rounds/canonical/`, regenerated by
`demo/build_cases.py`) — not a live re-run. TP/TN = correct, FP/FN = wrong. The 20 cases
are chosen so the failure modes are unmistakable, and they mirror the journal table
(`docs/RESULTS.md`):

**KorMMP journal table (5 seeds, n=500):** MiLTL-Cascade **AUROC 0.965 / F1 0.939** ·
Bllossom-B3 (8B) 0.684 · hf-encoder 0.641 · cnn-bilstm 0.658 · tree 0.638 · lexical 0.407 ·
Wave-Seq (audio-only) 0.508 · MiLTL-Dual (naive fusion) 0.561.

| Bundled case | Ground truth | What the record shows |
|---|---|---|
| Bank / card / hospital … consultations ×8 (`synth_hb_*`) | benign | **hf / tree / cnn — and usually the 8B Bllossom-B3 — FALSE-POSITIVE** — corpus classifiers learned "finance + authority words = scam". MiLTL = benign (TN). |
| Family / institution / loan … grooming ×7 (`synth_hh_*`) | harm | scam-word-free grooming (low density): the **8B Bllossom-B3 LLM misses every one (FN)**, audio-only Wave-Seq misses several. MiLTL catches them on arousal (E) + cross-modal contradiction (XM). |
| Real FSS recordings ×3 (`fssaud_*`, raw ASR) | harm | genuine noisy phone audio; includes a **Bllossom-B3 FN**, MiLTL TP. |
| Prosecutor / police impersonation ×2 (`korccvi:*`) | harm | KorCCViD home turf — **every legacy detector is correct here**, which is exactly why KorCCViD alone is a misleading benchmark. |

The pattern: legacy corpus classifiers look **flawless on their own corpus** (KorCCViD) and
break **in both directions** on the decorrelated hard slices — false-positive on legitimate
finance calls, and (for the LLM / audio-only) false-negative on vocabulary-free phishing.
That inversion is the KorMMP thesis (docs/BENCHMARK.md), shown from real results.

**Custom input** (pasted text / uploaded audio) has no recorded ground truth, so the panel
instead runs a couple of **live** budget detectors on your input — the lexical
keyword-density proxy (τ=2.0) and, if `pip install catboost` is present, the frozen
KorCCViD CatBoost tree (`artifacts/frozen/korccvid/tree/tree.pkl`) — next to MiLTL's band.

## Honest scope (read before quoting numbers)

- **Bundled-case verdicts are authoritative; the live trace is illustrative.** The
  per-detector verdicts on a bundled case are quoted verbatim from the canonical run. The
  animated MiLTL trace, however, re-computes Gate-1 with a **demo-fit Calib** (see below),
  so its `p1` shows the *mechanism* and may differ from the canonical `p1`. The recorded
  MiLTL-Cascade verdict in the table is the benchmark truth.
- **Calibration.** The demo **auto-loads the release Calib**
  (`artifacts/models/calib.release.json`, shipped in this repo) — the same calibration the
  canonical benchmark used. If the file is absent it falls back to a demo-fit Calib and
  says so in the startup log / UI.
- **Prosody: real for uploads, simulated otherwise.** Audio you upload uses **real**
  prosody (telephone-band + μ-law equalized, the same front end as the benchmark). For
  bundled/pasted text without audio, the labeled warm/cold/cold-pressure profiles
  synthesize only the features the affect mapping consumes, so XM stays observable — real
  call audio is not bundled (licensing — docs/DATA_ACCESS.md).
- **Auto-ASR quality depends on real speech and model size.** `--asr-model` selects the
  faster-whisper size (`tiny`/`base`/`small`…). A `.wav` of actual Korean speech
  transcribes well; non-speech audio yields an empty transcript → "undecidable".
- **The released SFT adapter is in this repo and auto-detected.** Run with
  `--gate2-model Qwen/Qwen2.5-1.5B-Instruct` and the demo attaches
  `artifacts/models/gate2_adapter_1.5b/` automatically — that is the exact canonical Gate-2
  (the L2 panel shows "SFT adapter"). On the default 0.5B backbone the adapter is skipped
  (base-model mismatch) and Gate-2 runs zero-shot — expect the paper's motivation to show
  up: a zero-shot judge can be fooled by scam-vocabulary-free grooming that Gate-1
  correctly escalates on prosody.
- **Rationale readability.** The escalation *decision* is fixed by Gate-2 scoring (a single
  forward pass); the displayed reason has two parts: a **deterministic, channel-grounded
  sentence** (always readable — built from the T/I/F/E/XM channels + transcript cues) and,
  on top, a **one-sentence SLM completion shown only if it passes a sanity filter**. This
  is deliberate: a 0.5B backbone cannot free-generate a reliable rationale (verified — it
  rambles or emits apology boilerplate), so on 0.5B you will usually see just the
  deterministic line. For an LLM-written sentence use `--gate2-model
  Qwen/Qwen2.5-1.5B-Instruct` (the canonical backbone + released SFT adapter).

## Files

- `miltl_demo.py` — single-file server + UI (stdlib HTTP/SSE; no web framework).
- `cases_canonical.json` — the 20 bundled cases: verbatim benchmark transcripts + the
  recorded per-detector verdicts. Sources: synthetic (self-authored, Apache-2.0), FSS
  (KOGL Type-1), KorCCViD (CC BY-NC-SA 4.0); no AI-Hub-derived transcripts.
- `build_cases.py` — regenerates `cases_canonical.json` from
  `artifacts/rounds/canonical/` (run after refreshing the benchmark artifacts).
- `edge_benchmark.py` — the budget benchmark behind the table above.
- `requirements-demo.txt` — numpy (L1); optional torch+transformers(+peft for the SFT
  adapter) (L2), catboost (live legacy tree), faster-whisper (wave-only ASR).
