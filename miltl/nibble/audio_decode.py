"""Audio decode seam — mp3/mp4/m4a/wav/pcm → mono float32 pcm (waveform path P2 entry point).

Turns FSS phishing audio (voice2 mp3, video mp4) into prosody_stream input. Compressed
formats go through an ffmpeg subprocess (edge/DGX standard), wav uses the stdlib, raw pcm
is read directly. Returns the same scale as prosody.read_pcm ([-1,1] mono). Requires numpy
(waveform path).

Design: MiLTL is natively multimodal (see design notes) — this seam is the front-end of
the waveform channel, symmetric with text.
"""
from __future__ import annotations

import io
import os
import shutil
import struct
import subprocess
import wave
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np

COMPRESSED = {".mp3", ".mp4", ".m4a", ".aac", ".ogg", ".wma", ".webm", ".opus", ".flac"}
# ffmpeg decode timeout (seconds) — blocks infinite hangs on corrupt/pathological files
# (prevents featurize stalls). Adjustable via env.
_FFMPEG_TIMEOUT = float(os.environ.get("FFMPEG_TIMEOUT", "90"))


def _from_int16(raw: bytes) -> np.ndarray:
    return (np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0)


def decode_wav(path) -> Tuple[np.ndarray, int]:
    """stdlib wave → (float32 [-1,1] mono, sr). path is a path or file-like (BytesIO). Multi-channel is averaged."""
    with wave.open(path if hasattr(path, "read") else str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        x = _from_int16(raw)
    elif sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"지원하지 않는 wav sampwidth: {sw}")
    if ch > 1:
        x = x[: len(x) // ch * ch].reshape(-1, ch).mean(axis=1)
    return x, sr


_FFMPEG_MISSING = (
    "No ffmpeg available for compressed audio (mp3/mp4/m4a/…). Install a pip-bundled ffmpeg "
    "with `pip install imageio-ffmpeg` (no system install needed), or a system ffmpeg, or "
    "convert the file to .wav/.pcm."
)


def _ffmpeg_exe() -> "str | None":
    """Locate an ffmpeg executable: system ffmpeg first, else the pip-bundled imageio-ffmpeg
    static binary (cross-platform, no system install) so the demo works out of the box."""
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def decode_ffmpeg(path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Decode compressed audio with ffmpeg → 16bit mono pcm (sr). Clear error if ffmpeg is missing."""
    ff = _ffmpeg_exe()
    if not ff:
        raise RuntimeError(_FFMPEG_MISSING)
    # -nostdin: keeps ffmpeg from blocking on stdin (infinite hang). timeout: blocks endless waits on corrupt files.
    cmd = [ff, "-nostdin", "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(sr), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True, timeout=_FFMPEG_TIMEOUT).stdout
    return _from_int16(out), sr


def _decode_ffmpeg_bytes(data: bytes, sr: int = 16000) -> np.ndarray:
    """Compressed audio bytes → 16bit mono pcm via ffmpeg stdin pipe. (zip members etc., without extraction)."""
    ff = _ffmpeg_exe()
    if not ff:
        raise RuntimeError(_FFMPEG_MISSING)
    cmd = [ff, "-v", "error", "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", str(sr), "-"]
    out = subprocess.run(cmd, input=data, capture_output=True, check=True, timeout=_FFMPEG_TIMEOUT).stdout
    return _from_int16(out)


def decode_bytes_to_pcm(data: bytes, ext: str, sr: int = 16000, pcm_bits: int = 16,
                        pcm_channels: int = 1) -> Tuple[np.ndarray, int]:
    """Audio **bytes** → (float32 [-1,1] mono, sr). For non-file sources such as zip members (no extraction)."""
    ext = ext.lower()
    if ext == ".wav":
        x, s = decode_wav(io.BytesIO(data))
        if s != sr:
            x = resample(x, s, sr)
        return x, sr
    if ext in (".pcm", ".raw"):
        from .prosody import read_pcm
        return read_pcm(data, bits=pcm_bits, channels=pcm_channels), sr
    if ext in COMPRESSED:
        return _decode_ffmpeg_bytes(data, sr), sr
    raise ValueError(f"지원하지 않는 오디오 확장자(바이트): {ext}")


def telephone_band(x: np.ndarray, sr: int, lo: float = 300.0, hi: float = 3400.0) -> np.ndarray:
    """Band-pass to the telephone band (300–3400Hz) — for domain control (FFT brick-wall).

    Removes the *recording-domain* difference between KsponSpeech (wideband studio) and FSS
    (narrowband telephone), so we can tell whether waveform-channel separability comes from
    the domain (bandwidth/codec) or from actual phishing prosody. Pure numpy (zero-phase)."""
    if len(x) < 2:
        return x
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    X[(f < lo) | (f > hi)] = 0.0
    return np.fft.irfft(X, n=len(x)).astype(np.float32)


def mulaw_codec(x: np.ndarray, mu: int = 255) -> np.ndarray:
    """G.711 μ-law companding round-trip (8bit quantization) — mimics the telephone-network codec
    signature. Pure numpy; deterministic apart from the quantization loss.

    Applying the same codec **uniformly to all audio** equalizes per-corpus codec differences
    (a leakage source) (docs/BENCHMARK.md).
    """
    if len(x) == 0:
        return x
    m = float(np.max(np.abs(x))) or 1.0
    xn = np.clip(x / m, -1.0, 1.0)
    comp = np.sign(xn) * np.log1p(mu * np.abs(xn)) / np.log1p(mu)     # compress
    q = np.round((comp + 1.0) / 2.0 * mu) / mu * 2.0 - 1.0            # 8bit quantization
    exp = (np.power(1.0 + mu, np.abs(q)) - 1.0) / mu                  # expand
    return (np.sign(q) * exp * m).astype(np.float32)


def equalize_channel(x: np.ndarray, sr: int, codec: bool = True) -> np.ndarray:
    """Fair audio channel equalization (docs/BENCHMARK.md): telephone-band band-pass + (optional) μ-law codec round-trip.

    Passes harm and benign audio through the **same band + codec**, forcing audio-only to
    discriminate on actual prosody rather than channel signature. Only meaningful when applied
    uniformly across all detectors and all calls (no asymmetric application)."""
    x = telephone_band(x, sr)
    return mulaw_codec(x) if codec else x


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Linear-interpolation resample (pure numpy). Aligns 48kHz stereo etc. → 16kHz mono pipeline."""
    if sr_in == sr_out or len(x) < 2:
        return x
    n = int(round(len(x) * sr_out / sr_in))
    if n <= 0:
        return np.array([], dtype=np.float32)
    return np.interp(np.linspace(0.0, len(x) - 1, n),
                     np.arange(len(x)), x).astype(np.float32)


def decode_to_pcm(path: str, sr: int = 16000, pcm_bits: int = 16,
                  pcm_channels: int = 1) -> Tuple[np.ndarray, int]:
    """Per-extension decode → (float32 [-1,1] mono, sr). wav is resampled if native sr≠target.
    pcm has no header, so sr/bits/channels must be assumed. `zip:<zip>!<member>` decodes the member without extracting."""
    if str(path).startswith("zip:"):
        zpath, member = str(path)[4:].rsplit("!", 1)     # zip:<path>!<member>
        with zipfile.ZipFile(zpath) as zf:
            data = zf.read(member)
        return decode_bytes_to_pcm(data, Path(member).suffix, sr, pcm_bits, pcm_channels)
    ext = Path(path).suffix.lower()
    if ext == ".wav":
        x, s = decode_wav(path)
        if s != sr:
            x = resample(x, s, sr)      # 48kHz etc. → target sr (prosody timing alignment)
        return x, sr
    if ext in (".pcm", ".raw"):
        from .prosody import read_pcm
        return read_pcm(Path(path).read_bytes(), bits=pcm_bits, channels=pcm_channels), sr
    if ext in COMPRESSED:
        return decode_ffmpeg(path, sr)
    raise ValueError(f"지원하지 않는 오디오 확장자: {ext}")


def _selftest() -> int:
    import tempfile, os
    # Synthetic sine-wave wav round-trip — verifies the wav path without ffmpeg
    sr = 16000
    t = np.arange(sr) / sr
    x = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        wp = os.path.join(d, "s.wav")
        with wave.open(wp, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes((x * 32767).astype("<i2").tobytes())
        y, sr2 = decode_to_pcm(wp)
        assert sr2 == sr and abs(len(y) - sr) <= 1, (len(y), sr2)
        assert abs(float(np.abs(y).max()) - 0.3) < 0.02, float(np.abs(y).max())
        # pcm path
        pp = os.path.join(d, "s.pcm")
        Path(pp).write_bytes((x * 32767).astype("<i2").tobytes())
        z, _ = decode_to_pcm(pp)
        assert abs(len(z) - sr) <= 1
    print("audio_decode selftest OK (wav/pcm). 압축(mp3/mp4)은 DGX ffmpeg 필요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
