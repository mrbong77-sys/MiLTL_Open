"""Native head input featurization (docs/ARCHITECTURE.md) — per-nibble (text_emb 768 + mel-spec).

- **Text**: klue-roberta-small mean-pool 768 (B1: native fine-tuning of a pretrained backbone). Offline selftest uses
  MockTextEncoder (deterministic hash), so no network is needed.
- **Audio**: mel-spec after telephone-band (300–3400Hz) control — removes the FSS (narrowband) ↔ benign (wideband) domain shortcut.
  mel is torch.stft + a hand-built mel filterbank (no torchaudio needed).
- **Alignment**: call audio is tiled into 8-second nibbles; each nibble is assigned the text of its interval (precise if timestamps exist, else uniform).

Observation envelope (docs/ARCHITECTURE.md): first L=26 nibbles (360 words ≈ 8s × 26). numpy output → the trainer batches tensors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

SR = 16000
SECONDS_PER_NIBBLE = 8.0
MAX_NIBBLES = 26                     # observation envelope (360 words ≈ 208s)


# ----------------------------------------------------------------- mel (no torchaudio needed)
def _mel_filterbank(n_mels: int, n_fft: int, sr: int, fmin: float = 300.0, fmax: float = 3400.0):
    """Telephone-band (300–3400Hz) triangular mel filterbank [n_mels, n_fft//2+1]. numpy."""
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    m_pts = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    hz_pts = mel2hz(m_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        for k in range(l, c):
            if 0 <= k < fb.shape[1] and c > l:
                fb[i - 1, k] = (k - l) / (c - l)
        for k in range(c, r):
            if 0 <= k < fb.shape[1] and r > c:
                fb[i - 1, k] = (r - k) / (r - c)
    return fb


def mel_spectrogram(pcm: np.ndarray, sr: int = SR, n_mels: int = 64,
                    n_fft: int = 512, hop: int = 160) -> np.ndarray:
    """pcm (float32 mono) → log-mel [n_mels, T]. Uses torch.stft when available, numpy FFT fallback otherwise."""
    if len(pcm) < n_fft:
        pcm = np.pad(pcm, (0, n_fft - len(pcm)))
    fb = _mel_filterbank(n_mels, n_fft, sr)
    try:
        import torch
        x = torch.from_numpy(np.ascontiguousarray(pcm)).float()
        win = torch.hann_window(n_fft)
        spec = torch.stft(x, n_fft=n_fft, hop_length=hop, window=win,
                          return_complex=True).abs() ** 2          # [F, T]
        mel = torch.from_numpy(fb) @ spec                          # [n_mels, T]
        return torch.log(mel + 1e-6).numpy().astype(np.float32)
    except Exception:
        # numpy STFT fallback
        w = np.hanning(n_fft).astype(np.float32)
        frames = [pcm[i:i + n_fft] for i in range(0, max(1, len(pcm) - n_fft), hop)]
        S = np.stack([np.abs(np.fft.rfft((f if len(f) == n_fft else
                     np.pad(f, (0, n_fft - len(f)))) * w)) ** 2 for f in frames], axis=1)
        mel = fb @ S
        return np.log(mel + 1e-6).astype(np.float32)


# ----------------------------------------------------------------- text encoder (pluggable)
class MockTextEncoder:
    """Deterministic mock for offline selftest — text hash → 768 vector (no network). Real training uses KlueRoberta."""

    dim = 768

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = abs(hash(t)) % (2 ** 31)
            rng = np.random.default_rng(h)
            out[i] = rng.standard_normal(self.dim).astype(np.float32)
        return out


class KlueRobertaEncoder:
    """klue/roberta-small mean-pool 768 (B1). Lazy load. frozen (default) or unfreeze (teacher, D1).

    For unfrozen training the trainer batch-forwards text — this wrapper is for inference/pre-embedding (frozen)."""

    def __init__(self, model_name: str = "klue/roberta-small", device: str = "cpu", max_len: int = 64):
        self.model_name = model_name
        self.device = device
        self.max_len = max_len
        self._tok = None
        self._model = None

    def _load(self):
        if self._model is None:
            from transformers import AutoTokenizer, AutoModel
            import torch
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
            self.dim = self._model.config.hidden_size

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self._load()
        import torch
        with torch.no_grad():
            enc = self._tok(list(texts), padding=True, truncation=True,
                            max_length=self.max_len, return_tensors="pt").to(self.device)
            out = self._model(**enc).last_hidden_state               # [B,T,H]
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)   # mean-pool
        return pooled.cpu().numpy().astype(np.float32)


# ----------------------------------------------------------------- nibble alignment + featurize
@dataclass
class NibbleFeatures:
    text_emb: np.ndarray     # [L, 768]
    mel: np.ndarray          # [L, n_mels, T]
    mask: np.ndarray         # [L]  (1=valid)
    n_valid: int


def _assign_text(utterances: Sequence[str], times: Optional[Sequence[Tuple[float, float]]],
                 n_nibbles: int, total_s: float) -> List[str]:
    """Assign utterances to per-nibble text. If times (per-utterance start,end) exist, assign precisely by midpoint; otherwise uniform."""
    buckets = [[] for _ in range(n_nibbles)]
    if times and len(times) == len(utterances):
        for u, (s, e) in zip(utterances, times):
            idx = min(n_nibbles - 1, int(((s + e) / 2.0) / SECONDS_PER_NIBBLE))
            buckets[max(0, idx)].append(u)
    else:                                                   # uniform distribution (no timestamps)
        if utterances:
            per = max(1, len(utterances) / n_nibbles)
            for j, u in enumerate(utterances):
                buckets[min(n_nibbles - 1, int(j / per))].append(u)
    return [" ".join(b) for b in buckets]


def featurize_call(pcm: Optional[np.ndarray], utterances: Sequence[str], text_enc,
                   times: Optional[Sequence[Tuple[float, float]]] = None,
                   sr: int = SR, n_mels: int = 64, max_nibbles: int = MAX_NIBBLES,
                   telephone: bool = True, words_per_nibble: int = 14) -> NibbleFeatures:
    """Call pcm + transcript → per-nibble (text_emb, mel, mask). Telephone-band control (removes domain shortcut).

    **text-only (pcm=None/empty)**: no audio, e.g. callcenter → mel=0 (modality-dropout training keeps this robust).
    The nibble count is determined by the text word count (÷ words_per_nibble).
    """
    texts, mel_p, mask_p, n = _tile(pcm, utterances, times, sr, n_mels, max_nibbles,
                                    telephone, words_per_nibble)
    emb = text_enc.encode(texts[:n]) if n else np.zeros((0, 768), np.float32)
    emb_p = np.zeros((max_nibbles, emb.shape[1] if emb.size else 768), np.float32)
    emb_p[:n] = emb
    return NibbleFeatures(emb_p, mel_p, mask_p, n)


def _tile(pcm, utterances, times, sr, n_mels, max_nibbles, telephone, words_per_nibble):
    """Shared tiling — 8-second-nibble-aligned text + mel (telephone-band control) + mask. Common to frozen/unfreeze.
    Returns: (texts[list, length max_nibbles], mel_p[L,n_mels,T], mask_p[L], n_valid)."""
    text_only = pcm is None or len(pcm) == 0
    if not text_only and telephone:
        from miltl.nibble.audio_decode import telephone_band
        pcm = telephone_band(pcm, sr)
    seg = int(sr * SECONDS_PER_NIBBLE)
    L = max_nibbles
    if text_only:
        n_words = sum(len(u.split()) for u in utterances)
        n = min(max_nibbles, max(1, int(np.ceil(n_words / max(1, words_per_nibble)))))
        texts = _assign_text(utterances, times, n, n * SECONDS_PER_NIBBLE)
        # ★ keep mel T **identical** to the audio path (actual mel frame count of 8s of zeros) — batch stack consistency
        T0 = mel_spectrogram(np.zeros(seg, np.float32), sr, n_mels).shape[1]
        mel_p = np.zeros((L, n_mels, T0), np.float32)
        mask_p = np.zeros(L, np.float32); mask_p[:n] = 1.0
        return texts + [""] * (L - n), mel_p, mask_p, n

    n = min(max_nibbles, max(1, int(np.ceil(len(pcm) / seg))))
    texts = _assign_text(utterances, times, n, len(pcm) / sr)
    mels, T_ref = [], None
    for i in range(n):
        chunk = pcm[i * seg:(i + 1) * seg]
        if len(chunk) < int(0.3 * sr):
            chunk = np.pad(chunk, (0, seg - len(chunk))) if len(chunk) else np.zeros(seg, np.float32)
        m = mel_spectrogram(chunk, sr, n_mels)
        T_ref = m.shape[1] if T_ref is None else T_ref
        if m.shape[1] != T_ref:
            m = (np.pad(m, ((0, 0), (0, T_ref - m.shape[1]))) if m.shape[1] < T_ref else m[:, :T_ref])
        mels.append(m)
    mel_arr = np.stack(mels, axis=0)
    mel_p = np.zeros((L, n_mels, mel_arr.shape[2]), np.float32); mel_p[:n] = mel_arr
    mask_p = np.zeros(L, np.float32); mask_p[:n] = 1.0
    return texts + [""] * (L - n), mel_p, mask_p, n


def featurize_call_tokens(pcm, utterances, tokenizer,
                          times=None, sr=SR, n_mels=64, max_nibbles=MAX_NIBBLES,
                          telephone=True, words_per_nibble=14, max_tok=64):
    """For unfreeze mode — returns **nibble tokens (ids/attn)** instead of text_emb (input for backbone fine-tuning).
    Returns: dict(text_ids[L,max_tok], text_attn[L,max_tok], mel[L,n_mels,T], mask[L], n_valid)."""
    texts, mel_p, mask_p, n = _tile(pcm, utterances, times, sr, n_mels, max_nibbles,
                                    telephone, words_per_nibble)
    ids, attn = tokenize_nibbles(texts, tokenizer, max_tok)
    return {"text_ids": ids, "text_attn": attn, "mel": mel_p, "mask": mask_p, "n_valid": n}


def tokenize_nibbles(texts, tokenizer, max_tok=64):
    """List of nibble texts → (ids[L,max_tok], attn[L,max_tok]) int32/float32."""
    ids = np.zeros((len(texts), max_tok), np.int64)
    attn = np.zeros((len(texts), max_tok), np.float32)
    for i, t in enumerate(texts):
        toks = tokenizer.encode(t or "", max_tok)
        ids[i, :len(toks)] = toks
        attn[i, :len(toks)] = 1.0
    return ids, attn


class MockTokenizer:
    """For offline selftest — word hash → vocab id (no network). Real training uses KlueRobertaTokenizer."""

    def __init__(self, vocab: int = 1000):
        self.vocab = vocab

    def encode(self, text: str, max_tok: int):
        ws = text.split()[:max_tok - 1] if text else []
        return [1] + [abs(hash(w)) % (self.vocab - 2) + 2 for w in ws]


class KlueRobertaTokenizer:
    """klue/roberta-small tokenizer wrapper (lazy). Input for unfreeze training."""

    def __init__(self, model_name: str = "klue/roberta-small"):
        self.model_name = model_name
        self._tok = None

    def encode(self, text: str, max_tok: int):
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
        return self._tok.encode(text or "", truncation=True, max_length=max_tok)


def _selftest() -> int:
    rng = np.random.default_rng(0)
    # Synthetic call: 20s sine + noise, 6 utterances
    t = np.arange(int(SR * 20)) / SR
    pcm = (0.2 * np.sin(2 * np.pi * 220 * t) + 0.02 * rng.standard_normal(len(t))).astype(np.float32)
    utts = ["안녕하세요 고객님", "대출 관련 안내드립니다", "지금 바로 입금하셔야",
            "계좌번호 알려주세요", "확인 부탁드립니다", "감사합니다"]
    enc = MockTextEncoder()
    nf = featurize_call(pcm, utts, enc, n_mels=64)
    assert nf.text_emb.shape == (MAX_NIBBLES, 768), nf.text_emb.shape
    assert nf.mel.shape[0] == MAX_NIBBLES and nf.mel.shape[1] == 64
    assert nf.mask.sum() == nf.n_valid
    print(f"[selftest] featurize OK · nibbles={nf.n_valid} · text_emb{nf.text_emb.shape} · "
          f"mel{nf.mel.shape} · mask합={nf.mask.sum():.0f}")
    # mel alone
    m = mel_spectrogram(pcm[:SR * 8], SR, 64)
    print(f"[selftest] mel_spectrogram {m.shape} (전화대역 mel, torch.stft) OK")
    print("[selftest] 네이티브 featurizer 동작(전화대역 통제·타임스탬프 정렬·관측봉투 패딩).")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
