#!/usr/bin/env python3
"""ASR transcript generation — attach text to audio-only data to complete dual-modal pairs.

For audio that has no transcript, run an injected ASR (`transcribe(pcm, sr)->str`) to create a
**sidecar `.txt`** (same stem, next to the audio). The existing dual adapters
(ksponspeech_dual / audio_dual / sample_voice) automatically pick these up via sibling-.txt
pairing → everything becomes dual-modal. Serves as a proof of the deployed ASR module and as
the data basis for the production routing simulation.

  # Inject a real ASR (e.g. the whisper wrapper in adapters/asr_adapter.py)
  scripts/dgx_run.sh --msg "generate ASR transcripts" python scripts/asr_transcribe.py \
     --root "data/raw/normal/<audio data>" --asr adapters.asr_adapter:transcribe

  python scripts/asr_transcribe.py --selftest    # verify the plumbing with a mock ASR
⚠️ Process both phishing and benign audio with the same ASR (symmetric errors). If a
ground-truth transcript exists, sidecar creation is skipped (--skip-existing).
"""
from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_AUDIO_EXT = (".wav", ".pcm", ".mp3", ".mp4", ".m4a", ".flac", ".aac", ".ogg", ".raw")


def _iter_audio(root: str):
    """Audio under root → path (loose file) or `zip:<zip>!<member>` URI (inside a zip). decode_to_pcm handles both.
    Audio inside zips (as in emotion_dialog_full / sample_voice) is included as ASR input too (ensures symmetric errors)."""
    for p in sorted(Path(root).rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in _AUDIO_EXT:
            yield str(p)
        elif ext == ".zip":
            try:
                zf = zipfile.ZipFile(p)
            except zipfile.BadZipFile:
                continue
            for zi in zf.infolist():
                if zi.is_dir():
                    continue
                if Path(zi.filename).suffix.lower() in _AUDIO_EXT:
                    yield f"zip:{p}!{zi.filename}"


def _selftest() -> int:
    import tempfile, os, wave
    import numpy as np
    from miltl.nibble.audio_decode import decode_to_pcm
    with tempfile.TemporaryDirectory() as d:
        wp = os.path.join(d, "x.wav")
        with wave.open(wp, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes((0.1 * np.sin(2 * np.pi * 200 * np.arange(16000) / 16000) * 32767)
                          .astype("<i2").tobytes())
        x, sr = decode_to_pcm(wp, sr=16000)
        txt = _mock_asr(x, sr)
        Path(wp).with_suffix(".txt").write_text(txt, encoding="utf-8")
        assert Path(wp).with_suffix(".txt").exists() and txt
    print(f"[selftest] OK — mock ASR transcript sidecar creation works ('{txt}'). Inject a real ASR with --asr.")
    return 0


def _mock_asr(x, sr) -> str:
    """Development mock — length-based dummy transcript (plumbing check). Replace with a real ASR."""
    secs = len(x) / sr if sr else 0
    return f"음성 전사 자리표시자 {secs:.0f}초"


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate ASR transcripts (complete audio→text pairs)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", default=None, help="Audio root (scanned recursively)")
    ap.add_argument("--audio-list", default=None,
                    help="File listing audio paths/URIs (one per line). When given, transcribe only this list "
                         "instead of scanning --root (use select_asr_audio.py to select all harm audio + a diverse benign sample).")
    ap.add_argument("--asr", default=None, help="module:fn  transcribe(pcm, sr)->str (real ASR)")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--skip-existing", action="store_true", help="Skip if a sibling .txt/.json already exists")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-suffix", default=".txt", help="Transcript sidecar extension (default .txt)")
    ap.add_argument("--mirror-dir", default="artifacts/asr",
                    help="Mirror location for sidecars of audio inside zips (cannot write next to them). Same convention as materialize.")
    ap.add_argument("--uniform", action="store_true",
                    help="Uniform ASR mode: re-transcribe ALL audio even when human transcripts exist → .asr.txt "
                         "(human transcripts are kept as ground truth). Equalizes text quality across classes (removes the quality shortcut).")
    args = ap.parse_args()

    if args.uniform:                                   # uniform mode: separate sidecar, no skipping
        args.out_suffix = ".asr.txt"
        args.skip_existing = False
        print("[uniform] re-transcribing all audio → .asr.txt (human transcripts preserved; quality equalized)")

    if args.selftest:
        return _selftest()
    if not args.root and not args.audio_list:
        ap.error("--root or --audio-list required (or --selftest)")

    from miltl.nibble.audio_decode import decode_to_pcm
    from miltl.baseline.asr_sidecar import asr_sidecar_path
    if args.asr:
        from miltl.nibble.integrations import load_callable
        asr = load_callable(args.asr)
        print(f"ASR: {args.asr}")
    else:
        asr = _mock_asr
        print("⚠️ --asr not given → mock ASR (placeholder). Inject a real ASR with --asr.")

    if args.audio_list:                               # explicit list input (all harm + diverse benign sample)
        auds = [ln.strip() for ln in Path(args.audio_list).read_text(encoding="utf-8").splitlines()
                if ln.strip()]
        src_desc = f"list {args.audio_list}"
    else:
        auds = list(_iter_audio(args.root))           # loose files + audio inside zips (zip: URIs)
        src_desc = f"root {args.root}"
    if args.limit and args.limit > 0:                 # only the first N (smoke/subset) — previously unwired (bug), now fixed
        auds = auds[:args.limit]
        src_desc += f" [limit {args.limit}]"
    total = len(auds)
    n, skip, fail, t0 = 0, 0, 0, time.time()
    n_zip = sum(1 for a in auds if a.startswith("zip:"))
    print(f"Transcribing {total} audio files ({n_zip} inside zips) · {src_desc}", flush=True)
    for i, item in enumerate(auds, 1):
        nm = Path(item.split("!", 1)[-1] if item.startswith("zip:") else item).name
        sib = asr_sidecar_path(item, args.out_suffix, args.mirror_dir)
        if args.skip_existing and sib.exists():
            skip += 1
            print(f"  [{i}/{total}] ⏭ skip (already exists) {nm}", flush=True)
            continue
        if args.skip_existing and not item.startswith("zip:"):   # loose file: skip if a human transcript (sibling) exists
            pp = Path(item)
            if pp.with_suffix(".json").exists() or pp.with_suffix(".trn").exists():
                skip += 1
                print(f"  [{i}/{total}] ⏭ skip (human transcript) {nm}", flush=True)
                continue
        t1 = time.time()
        try:
            x, sr = decode_to_pcm(item, sr=args.sr)
            txt = (asr(x, sr) or "").strip()
            dt = time.time() - t1
            if not txt:                                           # empty transcript → no sidecar written (retried on re-run)
                fail += 1
                print(f"  [{i}/{total}] ⚠️ empty transcript {nm} ({dt:.1f}s) — sidecar not written", flush=True)
            else:
                sib.parent.mkdir(parents=True, exist_ok=True)     # ensure the mirror directory exists
                sib.write_text(txt, encoding="utf-8")
                n += 1
                print(f"  [{i}/{total}] ✓ {nm} · {len(txt.split())} words · {dt:.1f}s "
                      f"(done {n}·failed {fail}·skipped {skip})", flush=True)
        except Exception as e:
            fail += 1
            print(f"  [{i}/{total}] ✗ failed {nm}: {type(e).__name__}: {str(e)[:60]}", flush=True)
        # Early abort if everything fails at the start (catches backend/CUDA miswiring before wasting hours)
        if n == 0 and fail >= 20:
            print(f"❌ first {fail} attempts failed in a row with 0 successes — suspected backend/device miswiring. Aborting.\n"
                  "   Check: ASR_BACKEND/ASR_DEVICE, the faster-whisper CUDA build, and the [asr_adapter] candidate failure reasons in the log.",
                  flush=True)
            return 3
    print(f"Done: transcribed {n} · skipped {skip} · failed {fail} → {args.out_suffix} sidecars created. "
          "Dual adapters auto-pair via sibling .txt files.")
    return 0 if (n > 0 or len(auds) == 0) else 4


if __name__ == "__main__":
    raise SystemExit(main())
