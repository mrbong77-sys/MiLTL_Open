"""ASR sidecar path convention — **must be shared** by writer (asr_transcribe) and reader (materialize).

loose audio:   saved next to it as <stem><suffix>   (e.g. call.wav → call.asr_light.txt)  ← existing convention (fss_audio_calls·test_asr_track)
zip:URI audio: mirrored at <mirror>/<stem>__<uri-hash8><suffix> (cannot write inside the zip)

Audio inside zips (emotion_dialog_full·sample_voice) uses `zip:<zip>!<member>` URIs, so a
sidecar cannot sit next to it → deterministic filename in a mirror directory. stem for
readability, hash8 for collision safety (distinguishes same stem across different zips).
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def asr_sidecar_path(audio_uri, suffix: str, mirror_dir: str = "artifacts/asr") -> Path:
    """Audio (path or zip:URI) → ASR sidecar path. suffix example: '.asr_light.txt'."""
    s = str(audio_uri)
    if s.startswith("zip:"):
        member = s.split("!", 1)[1] if "!" in s else s[4:]
        stem = Path(member).stem
        h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]
        return Path(mirror_dir) / f"{stem}__{h}{suffix}"
    p = Path(s)
    return p.with_name(p.stem + suffix)           # x.wav → x<suffix> (stem convention)
