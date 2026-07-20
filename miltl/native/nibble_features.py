"""Channel extractor input featurization (docs/ARCHITECTURE.md) — call → per-nibble (prosody, text, speech-act).

Frontend of the channel bottleneck. For each 8-second nibble:
  · prosody[18]  : prosodic features (miltl.nibble.prosody.ProsodyFeatures) — acoustic evidence for E/AVD/F/I/T
  · text[768]    : klue-roberta mean-pool — semantic component for F/I/T (offline = MockTextEncoder)
  · speech_act[4]: directive/urgency/threat/subversion (miltl.native.speech_act; lexical proxy if absent)
Observation envelope L=26. No audio → prosody=0 (text-only). Telephone-band control (domain shortcut).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from miltl.native.features import _assign_text, MockTextEncoder, SR, SECONDS_PER_NIBBLE, MAX_NIBBLES
from miltl.nibble.prosody import prosody_stream, segment_prosody

_PROS_KEYS = ["voiced_ratio", "f0_mean", "f0_std", "f0_slope", "energy_mean", "energy_std",
              "energy_slope", "zcr_mean", "rate_proxy", "jitter", "shimmer", "hnr_mean",
              "spectral_centroid", "spectral_tilt", "f0_range", "pause_ratio", "pause_rate", "mean_pause_s"]
PROS_DIM = len(_PROS_KEYS)
SA_AXES = ("directive", "urgency", "threat", "subversion")


@dataclass
class NibbleChannelInput:
    prosody: np.ndarray   # [L, 18]
    text: np.ndarray      # [L, 768]
    speech_act: np.ndarray  # [L, 4]  directive/urgency/threat/subversion
    mask: np.ndarray      # [L]
    n_valid: int
    warmth: np.ndarray = None  # [L] affinity/prosocial lexicality (cross-modal I term)


def _pros_vec(pf) -> np.ndarray:
    d = pf.as_dict()
    return np.array([float(d.get(k, 0.0)) for k in _PROS_KEYS], np.float32)


def _core(pcm, utterances, times, sr, max_nibbles, telephone, codec_equalize=False):
    """Shared frontend: (pros_list, texts, n, total_s). prosody_stream runs once."""
    if pcm is not None and len(pcm):
        if codec_equalize:                                  # docs/BENCHMARK.md: telephone-band + uniform μ-law equalization
            from miltl.nibble.audio_decode import equalize_channel
            pcm = equalize_channel(pcm, sr, codec=True)
        elif telephone:
            from miltl.nibble.audio_decode import telephone_band
            pcm = telephone_band(pcm, sr)
        pros_list = prosody_stream(pcm, sr, SECONDS_PER_NIBBLE)[:max_nibbles]
        n = max(1, len(pros_list))
        total_s = len(pcm) / sr
    else:
        pros_list = []
        nwords = sum(len(u.split()) for u in utterances)
        n = min(max_nibbles, max(1, int(np.ceil(nwords / 14))))
        total_s = n * SECONDS_PER_NIBBLE
    texts = _assign_text(utterances, times, n, total_s)
    return pros_list, texts, n, total_s


def _build_input(pros_list, texts, n, text_enc, sa_clf, max_nibbles) -> NibbleChannelInput:
    L = max_nibbles
    pros = np.zeros((L, PROS_DIM), np.float32)
    for i in range(min(n, len(pros_list))):
        pros[i] = _pros_vec(pros_list[i])
    emb = text_enc.encode(texts[:n]) if n else np.zeros((0, 768), np.float32)
    text = np.zeros((L, emb.shape[1] if emb.size else 768), np.float32); text[:n] = emb
    sa = np.zeros((L, 4), np.float32)
    warmth = np.zeros(L, np.float32)
    from miltl.native.channel_teacher import _lex, _WARM
    for i in range(n):
        sa[i] = _speech_act(texts[i], sa_clf)
        warmth[i] = _lex(texts[i], _WARM)
    mask = np.zeros(L, np.float32); mask[:n] = 1.0
    return NibbleChannelInput(pros, text, sa, mask, n, warmth)


def featurize_channels(pcm, utterances, text_enc=None, sa_clf=None, times=None,
                       sr: int = SR, max_nibbles: int = MAX_NIBBLES, telephone: bool = True,
                       codec_equalize: bool = False):
    """Call → NibbleChannelInput. text_enc=None→Mock, sa_clf=None→lexical proxy.
    codec_equalize=True (docs/BENCHMARK.md): telephone-band + uniform μ-law equalization (fair audio)."""
    text_enc = text_enc or MockTextEncoder()
    pros_list, texts, n, _ = _core(pcm, utterances, times, sr, max_nibbles, telephone, codec_equalize)
    return _build_input(pros_list, texts, n, text_enc, sa_clf, max_nibbles)


# Channel targets (T,I,F,E) are produced by the neutrosophic/affect calibration (docs/ARCHITECTURE.md).
# featurize(raw) → fit_calib(benign) → channel_calib.channels(nci, cal). (Single-call target formula deprecated)


def _speech_act(text: str, sa_clf) -> np.ndarray:
    if sa_clf is not None:
        d = sa_clf.predict(text)
        return np.array([d[a] for a in SA_AXES], np.float32)
    # Lexical proxy (bootstrap before extractor training)
    from miltl.native.channel_teacher import _lex, _DIRECTIVE, _URGENCY, _THREAT, _SCAM
    return np.array([_lex(text, _DIRECTIVE), _lex(text, _URGENCY),
                     _lex(text, _THREAT), _lex(text, _SCAM)], np.float32)


def _selftest() -> int:
    rng = np.random.default_rng(0)
    t = np.arange(int(SR * 18)) / SR
    pcm = (0.15 * np.sin(2 * np.pi * 200 * t) + 0.02 * rng.standard_normal(len(t))).astype(np.float32)
    utts = ["고객님 안녕하세요", "지금 즉시 계좌 이체 하세요", "검찰 수사 관련입니다", "확인 부탁드립니다"]
    nf = featurize_channels(pcm, utts)
    assert nf.prosody.shape == (MAX_NIBBLES, PROS_DIM), nf.prosody.shape
    assert nf.text.shape == (MAX_NIBBLES, 768) and nf.speech_act.shape == (MAX_NIBBLES, 4)
    print(f"[selftest] featurize OK · prosody{nf.prosody.shape} text{nf.text.shape} sa{nf.speech_act.shape} "
          f"· nibbles={nf.n_valid}")
    # speech-act lexical proxy: the threat axis of the 'threat' sentence nibble > 0
    print(f"[selftest] speech_act[유효]=\n{np.round(nf.speech_act[:nf.n_valid],2)}")
    assert nf.warmth is not None and nf.warmth.shape == (MAX_NIBBLES,)
    print(f"[selftest] warmth[유효]={np.round(nf.warmth[:nf.n_valid],2)} (교차모달 I 항)")
    print("[selftest] 니블 채널 featurizer 동작(prosody+text+speech-act+warmth). 타깃=channel_calib(별도). 실행=DGX.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
