#!/usr/bin/env python3
"""
miltl_demo.py — standalone MiLTL demo: CPU-only, no GPU, single machine, web UI.

Experience the full MiLTL cascade (Gate-1 → banding → Gate-2) interactively on an
ordinary laptop, with live per-nibble channel traces, gate activation lamps, stage
latencies, and CPU/RAM occupancy — the evidence trail for the "runs on mobile-class
hardware" claim.

  L1 (Gate-1)  pure numpy. The canonical scorer is the deterministic analytic rule
               risk = E − 2·T + I + XM over the calibrated neutrosophic channels —
               zero learned parameters at inference, no torch required.
  L2 (Gate-2)  optional. The repo's real Gate2SLM (Qwen2.5 Instruct) loaded on CPU in
               fp32. Scoring is a SINGLE forward pass (yes/no token log-prob families),
               which is why it is laptop-feasible; the generated XAI rationale is
               available on demand.

Three ways to feed it:
  1. Bundled canonical cases — transcripts quoted verbatim from the 5-seed canonical
     benchmark, each carrying the REAL per-detector verdicts recorded in that run, so
     "why legacy detectors fail and MiLTL succeeds" is shown from actual results.
  2. Paste any Korean transcript (text-only path).
  3. Upload a .wav / .mp3 / .mp4 / .m4a — real prosody is extracted (compressed formats via the
     pip-bundled imageio-ffmpeg, no system install); if you
     give no transcript, it is auto-transcribed (faster-whisper, if installed).

Run (from the repo root):
    pip install -r demo/requirements-demo.txt
    python demo/miltl_demo.py                     # web UI at http://localhost:7861
    python demo/miltl_demo.py --no-gate2          # L1-only (numpy)
    python demo/miltl_demo.py --gate2-model Qwen/Qwen2.5-1.5B-Instruct
    python demo/miltl_demo.py --selftest          # terminal sanity check, no UI

Honest-scope notes (also shown in the UI):
  * Bundled-case verdicts are the AUTHORITATIVE canonical benchmark records. The live
    MiLTL trace re-computes the mechanism with a DEMO-fit Calib and (for text/simulated
    inputs) a simulated prosody profile, so its p1 illustrates the mechanism and need
    not equal the canonical p1. Drop in the released Calib with --calib.
  * Real audio you upload uses REAL prosody. Everything else uses labeled simulated
    prosody, since real call audio is not bundled (licensing — docs/DATA_ACCESS.md).
  * Gate-2 runs the base Instruct model zero-shot until the LoRA adapter is released
    (docs/MODELS.md); the prompt/scoring path is the repo's real Gate2SLM code.
  * Korean-only. Channel lexicons, calibration, Gate-2 prompts, and auto-ASR (language=ko)
    are all Korean-based; English/other-language input is not judged correctly.
  * Rationale: the decision is fixed by scoring; the reason line is a deterministic,
    channel-grounded sentence (always readable) with a tightly-constrained one-sentence SLM
    completion added only if it passes a sanity filter. Small backbones (0.5B) cannot
    free-generate a reliable rationale — the 1.5B backbone (or the forthcoming SFT adapter)
    is recommended for an LLM-written one.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from miltl.native.nibble_features import featurize_channels, NibbleChannelInput, MAX_NIBBLES  # noqa: E402
from miltl.native.channel_calib import (  # noqa: E402
    Calib, fit_calib, channels5, avd_from_z, evidence, _IX, PROS_KEYS,
)
from miltl.native.explain import explain_decision  # noqa: E402

# ── Canonical operating points (docs/ARCHITECTURE.md §4) ────────────────────
TAU_LOW, TAU_HIGH = 0.40, 0.90
W_T, W_XM = 2.0, 1.0
SCORE_SCALE = 0.5           # p1 = sigmoid(risk / SCORE_SCALE)
W_F_TEXT, F0_TEXT = 2.0, 0.5  # transcript-only lexical-F recentring compensation
ANCHOR_AUDIO, ANCHOR_TEXT = 80, 12
NIBBLE_SEC = 8.0            # one observation = 8 s of audio (matches SECONDS_PER_NIBBLE)
# Decision deadline = the canonical observation envelope. featurize_channels already caps MiLTL
# scoring at MAX_NIBBLES, so a call is *decided* within MAX_NIBBLES × 8 s regardless of its true
# length (an in-call early-warning detector does not wait for a 37-minute call to end). Overridable
# via --max-nibbles, but never beyond MAX_NIBBLES (scoring cannot see past the envelope anyway).
DECISION_NIBBLES = MAX_NIBBLES              # 26 nibbles = 208 s ≈ 3.5 min (canonical)

CASES_FILE = Path(__file__).resolve().parent / "cases_canonical.json"

# Release-weight auto-detection (docs/MODELS.md). Drop these in and the demo picks them up with
# no flags: release Calib (JSON preferred, else extracted from channel_extractors.pt) and the
# Gate-2 LoRA SFT adapter. All paths are white-listed in .gitignore.
_MODELS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "models"
_CALIB_JSON = _MODELS_DIR / "calib.release.json"
_CALIB_CKPT = _MODELS_DIR / "channel_extractors.pt"
_GATE2_ADAPTER_DIR = _MODELS_DIR / "gate2_adapter_1.5b"


def _load_release_calib(explicit: "str | None"):
    """Resolve the release Calib without forcing torch on the L1-only path.
    Order: --calib JSON → artifacts/models/calib.release.json → channel_extractors.pt['calib']."""
    path = explicit or (str(_CALIB_JSON) if _CALIB_JSON.exists() else None)
    if path:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return Calib.from_dict(d), f"loaded from {path} (release Calib)"
    if _CALIB_CKPT.exists():
        try:
            import torch
            d = torch.load(str(_CALIB_CKPT), map_location="cpu")
            if isinstance(d, dict) and "calib" in d:
                return Calib.from_dict(d["calib"]), f"loaded from {_CALIB_CKPT.name} (release Calib)"
        except Exception as e:  # noqa: BLE001
            print(f"[calib] {_CALIB_CKPT.name} present but not loadable ({type(e).__name__}); "
                  f"export a JSON with Calib.to_dict → {_CALIB_JSON.name}, or install torch.", flush=True)
    return None, None


# ── Simulated prosody profiles (demo-only, labeled) ─────────────────────────
# Same synthesis pattern as channel_calib._selftest: only the features consumed by
# avd_from_z matter. "warm" ≈ benign consultation voice; "cold" ≈ scripted scammer
# voice (low valence via positive spectral tilt, narrow f0 range) with a mild
# harm-ramp (arousal rising toward the end); "cold-pressure" adds arousal from start.
_PROFILES = {
    "warm":          dict(energy=0.045, f0=185.0, f0_sd=6.0, f0_range=55.0, tilt=-1.2,
                          hnr=8.0, pause=0.18, rate=0.0, ramp=0.0),
    "cold":          dict(energy=0.055, f0=172.0, f0_sd=3.0, f0_range=16.0, tilt=1.1,
                          hnr=3.0, pause=0.30, rate=0.2, ramp=1.0),
    "cold-pressure": dict(energy=0.110, f0=228.0, f0_sd=4.0, f0_range=22.0, tilt=1.3,
                          hnr=2.5, pause=0.12, rate=0.6, ramp=1.5),
}


def sim_prosody(n_valid: int, L: int, profile: str, rng: np.random.Generator) -> np.ndarray:
    p = _PROFILES[profile]
    pros = np.zeros((L, len(PROS_KEYS)), np.float32)
    n = n_valid
    e0 = p["energy"] * float(rng.normal(1.0, 0.12))
    f0 = p["f0"] + float(rng.normal(0.0, 8.0))
    ramp = np.linspace(0.0, p["ramp"], n)
    pros[:n, _IX["energy_mean"]] = e0 * (1.0 + 0.8 * ramp) + 0.006 * rng.standard_normal(n)
    pros[:n, _IX["f0_mean"]] = f0 * (1.0 + 0.12 * ramp) + p["f0_sd"] * rng.standard_normal(n)
    pros[:n, _IX["f0_range"]] = p["f0_range"] * (1.0 + 0.3 * ramp) + 3.0 * rng.standard_normal(n)
    pros[:n, _IX["spectral_tilt"]] = p["tilt"] + 0.15 * rng.standard_normal(n)
    pros[:n, _IX["hnr_mean"]] = p["hnr"] + 0.6 * rng.standard_normal(n)
    pros[:n, _IX["pause_ratio"]] = p["pause"] + 0.03 * rng.standard_normal(n)
    pros[:n, _IX["rate_proxy"]] = p["rate"] * (1.0 + 0.5 * ramp) + 0.06 * rng.standard_normal(n)
    return pros


# ── L1 engine (pure numpy) ──────────────────────────────────────────────────
class L1Engine:
    """Gate-1 analytic path: featurize → calibrated channels [L,5] → risk → p1 → band."""

    def __init__(self, calib_path: str | None = None, seed: int = 20260720):
        self.rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        rel_calib, rel_src = _load_release_calib(calib_path)
        if rel_calib is not None:
            self.cal_audio = rel_calib
            self.cal_text = rel_calib
            self.calib_src = rel_src
        else:
            self.cal_audio = self._fit_audio_calib()
            self.cal_text = self._fit_text_calib()
            self.calib_src = ("demo-fit: simulated warm-benign prosody (audio mode) + "
                              "bundled synthetic benign transcripts (text mode) — NOT the release Calib")
        self.calib_fit_s = time.perf_counter() - t0

    def _fit_audio_calib(self) -> Calib:
        benign = []
        for t in _benign_texts_for_calib()[:30]:
            nci = self._nci_text(t.split())
            nci = self._with_prosody(nci, "warm")
            benign.append(nci)
        return fit_calib(benign)

    def _fit_text_calib(self) -> Calib:
        benign = [self._nci_text(t.split()) for t in _benign_texts_for_calib()]
        return fit_calib(benign)

    def _nci_text(self, words) -> NibbleChannelInput:
        return featurize_channels(None, words)   # text_enc=None → Mock (text768 unused in calib mode)

    def _with_prosody(self, nci: NibbleChannelInput, profile: str) -> NibbleChannelInput:
        pros = sim_prosody(nci.n_valid, len(nci.mask), profile, self.rng)
        return NibbleChannelInput(pros, nci.text, nci.speech_act, nci.mask, nci.n_valid, nci.warmth)

    def featurize(self, transcript: str, prosody: str):
        """Text (+ optionally simulated prosody). prosody ∈ {'none','warm','cold','cold-pressure'}."""
        t0 = time.perf_counter()
        nci = self._nci_text(transcript.split())
        t_feat = time.perf_counter() - t0
        has_audio = prosody != "none"
        t_sim = 0.0
        if has_audio:
            t1 = time.perf_counter()
            nci = self._with_prosody(nci, prosody)
            t_sim = time.perf_counter() - t1
        return nci, has_audio, {"featurize_ms": t_feat * 1e3, "prosody_sim_ms": t_sim * 1e3,
                                "prosody_kind": prosody, "prosody_real": False}

    def featurize_real_audio(self, transcript: str, pcm: np.ndarray):
        """REAL prosody path — pcm from an uploaded audio file (no simulation)."""
        t0 = time.perf_counter()
        nci = featurize_channels(pcm, transcript.split(), text_enc=None, codec_equalize=True)
        dt = (time.perf_counter() - t0) * 1e3
        return nci, True, {"featurize_ms": dt, "prosody_sim_ms": 0.0,
                           "prosody_kind": "real-audio", "prosody_real": True}

    def score_prefix(self, nci: NibbleChannelInput, has_audio: bool, k: int):
        cal = self.cal_audio if has_audio else self.cal_text
        t0 = time.perf_counter()
        mask = nci.mask.copy()
        mask[k:] = 0.0
        sub = NibbleChannelInput(nci.prosody, nci.text, nci.speech_act, mask,
                                 min(k, nci.n_valid), nci.warmth)
        ch = channels5(sub, cal)
        m = mask > 0.5
        mean = ch[m].mean(axis=0) if m.any() else np.zeros(5, np.float32)
        T, I, F, E, XM = (float(x) for x in mean)
        z = cal.zfeat(sub.prosody)
        avd = avd_from_z(z)
        w = sub.warmth if sub.warmth is not None else np.zeros(len(mask), np.float32)
        ev = evidence(avd, sub.speech_act, w, z[:, _IX["pause_ratio"]])
        V = float(avd[m, 1].mean()) if m.any() else 0.5
        warmth = float(w[m].mean()) if m.any() else 0.0
        xm_ev = float(ev["XM"][m].mean()) if m.any() else 0.0
        risk = E - W_T * T + I + W_XM * xm_ev
        if not has_audio:
            risk += W_F_TEXT * (F - F0_TEXT)
        p1 = float(1.0 / (1.0 + np.exp(-risk / SCORE_SCALE)))
        band = "benign" if p1 <= TAU_LOW else ("harm" if p1 >= TAU_HIGH else "escalate")
        lat_ms = (time.perf_counter() - t0) * 1e3
        kk = k - 1
        valid = kk < len(ch) and mask[kk] > 0.5
        nib = ch[kk] if valid else np.zeros(5, np.float32)
        # Per-nibble XM decomposition (the two modalities whose mismatch IS XM):
        #   warmth = warm-word density (TEXT) ; V = voice valence (PROSODY) → cold = 1−V.
        warmth_k = float(w[kk]) if valid else 0.0
        V_k = float(avd[kk, 1]) if valid else 0.5
        D_k = float(avd[kk, 2]) if valid else 0.5
        return {
            "k": k,
            "nibble": {c: round(float(v), 4) for c, v in zip("TIFEX", nib)},
            "decomp": {"warmth": round(warmth_k, 4), "V": round(V_k, 4),
                       "cold": round(1.0 - V_k, 4), "D": round(D_k, 4)},   # XM inputs for this nibble
            "mean": {"T": T, "I": I, "F": F, "E": E, "XM": xm_ev,
                     "V": V, "cold": 1.0 - V, "warmth": warmth},
            "risk": round(risk, 4), "p1": round(p1, 4), "band": band,
            "l1_ms": round(lat_ms, 2),
        }

    def anchor_ok(self, transcript: str, has_audio: bool):
        n = len(transcript.split())
        anchor = ANCHOR_AUDIO if has_audio else ANCHOR_TEXT
        return n >= anchor, n, anchor


# ── runtime detection (CPU vs GPU) ─────────────────────────────────────────
_RUNTIME: dict = {}


def _detect_runtime() -> dict:
    """Detect the inference device once: {'device','gpu','detail'}. torch-free safe."""
    if _RUNTIME:
        return _RUNTIME
    device, gpu, detail = "cpu", False, "CPU only (torch not installed)"
    try:
        import torch
        if torch.cuda.is_available():
            device, gpu = "cuda", True
            try:
                detail = f"CUDA: {torch.cuda.get_device_name(0)}"
            except Exception:  # noqa: BLE001
                detail = "CUDA GPU"
        elif getattr(getattr(torch, "backends", None), "mps", None) is not None and \
                torch.backends.mps.is_available():
            device, gpu, detail = "mps", True, "Apple Metal (MPS) GPU"
        else:
            detail = "CPU only (no CUDA/MPS)"
    except Exception:  # noqa: BLE001
        pass
    _RUNTIME.update({"device": device, "gpu": gpu, "detail": detail})
    return _RUNTIME


# ── L2 (Gate-2 SLM) — real Gate2SLM; GPU when available, else CPU fp32 ──────
class L2Runtime:
    def __init__(self, model_name: str, threads: int = 0, adapter_path: "str | None" = None):
        self.model_name = model_name
        self.threads = threads
        self.adapter_path = adapter_path      # LoRA SFT adapter dir (release); None → zero-shot base
        self.adapter_on = False               # set True once the adapter actually attaches
        self.status = "not_loaded"
        self.detail = ""
        self.load_s = None
        self.device = _detect_runtime()["device"]
        self._g2 = None
        self._lock = threading.Lock()

    def start_loading(self):
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        with self._lock:
            if self.status in ("loading", "ready"):
                return
            self.status = "loading"
        t0 = time.perf_counter()
        try:
            import torch
            if self.threads:
                torch.set_num_threads(self.threads)
            from adapters.baselines.gate2_slm import Gate2SLM
            rt = _detect_runtime()
            _dev = rt["device"]
            _POS = ["예", " 예", "네", " 네", "위험", " 위험", "유해", "Yes", " Yes"]
            _NEG = ["아니", " 아니", "아니오", " 아니오", "정상", " 정상", "안전", "No", " No"]
            _adapter = self.adapter_path
            _rt_self = self

            class _DemoGate2(Gate2SLM):
                def _build(self):
                    torch_ = self._torch
                    self._tok = self._AutoTok.from_pretrained(self.model_name)
                    if _dev == "cuda":            # GPU: fp16, auto-placed
                        self._model = self._AutoLM.from_pretrained(
                            self.model_name, torch_dtype=torch_.float16, device_map="auto")
                    elif _dev == "mps":           # Apple Metal: fp16 on mps
                        self._model = self._AutoLM.from_pretrained(
                            self.model_name, torch_dtype=torch_.float16).to("mps")
                    else:                          # CPU: fp32
                        self._model = self._AutoLM.from_pretrained(
                            self.model_name, torch_dtype=torch_.float32)
                    if _adapter:                  # attach the release LoRA SFT adapter
                        from peft import PeftModel
                        self._model = PeftModel.from_pretrained(self._model, _adapter)
                        _rt_self.adapter_on = True
                    self._model.eval()
                    self._pos_ids = self._first_ids(_POS)
                    self._neg_ids = self._first_ids(_NEG)

            g2 = _DemoGate2(model_name=self.model_name)
            g2._ensure()
            g2._build()
            self._g2 = g2
            self.load_s = round(time.perf_counter() - t0, 1)
            self.status = "ready"
            _prec = "fp16" if _dev in ("cuda", "mps") else "fp32"
            _sft = " · SFT adapter" if self.adapter_on else " · zero-shot base"
            self.detail = f"{self.model_name} · {_dev} {_prec}{_sft} · load {self.load_s}s"
        except Exception as e:  # noqa: BLE001
            import traceback
            self.status = "unavailable"
            self.detail = f"{type(e).__name__}: {e}"
            print(f"\n[gate2] LOAD FAILED — {self.detail}", flush=True)
            traceback.print_exc()
            print("[gate2] hint: the demo process needs torch + transformers "
                  "(+ peft for the SFT adapter) in the SAME environment. Check with: "
                  "python -c 'import torch,transformers,peft'\n", flush=True)

    def score(self, transcript: str, diag: dict):
        t0 = time.perf_counter()
        p2 = float(self._g2.score_channels(transcript, diag))
        return p2, (time.perf_counter() - t0)

    def concise_rationale(self, transcript: str, diag: dict, verdict: str):
        """Readable rationale robust to backbone size.

        The decision is already fixed by score_channels; here we only need a short reason.
        Small backbones (0.5B) cannot free-generate a reliable one (they ramble or emit
        out-of-menu tokens — verified), so the reliable line is a DETERMINISTIC,
        channel-grounded sentence. A tightly-constrained one-sentence SLM completion is
        added on top ONLY if it passes a sanity filter (usable on 1.5B+).
        """
        det = deterministic_rationale(diag, verdict, transcript)
        t0 = time.perf_counter()
        slm = None
        try:
            slm = self._gen_one_sentence(transcript, verdict)
        except Exception:  # noqa: BLE001
            slm = None
        return {"reason": det, "slm": slm, "latency_s": round(time.perf_counter() - t0, 1)}

    def _gen_one_sentence(self, transcript: str, verdict: str):
        torch = self._g2._torch
        vlabel = "피싱(위험)" if verdict == "harm" else "정상"
        sys_p = ("당신은 보이스피싱 판정 안내원입니다. 이미 내려진 판정의 이유를, 통화에서 실제로 나타난 "
                 "정황을 근거로 쉬운 한국어 한 문장(50자 이내)으로만 설명하세요. 사과·서론·반복 없이 이유만 쓰세요.")
        msgs = [{"role": "system", "content": sys_p},
                {"role": "user", "content": f"판정: {vlabel}\n통화 전사:\n{(transcript or '')[:900]}\n\n이유(한 문장):"}]
        tok = self._g2._tok
        try:
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:  # noqa: BLE001
            prompt = sys_p + "\n" + msgs[1]["content"]
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1200).to(self._g2._model.device)
        with torch.no_grad():
            out = self._g2._model.generate(**enc, max_new_tokens=60, do_sample=False,
                                           repetition_penalty=1.2, no_repeat_ngram_size=3,
                                           pad_token_id=tok.eos_token_id)
        raw = tok.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        return _sanitize_sentence(raw, transcript)


# ── rationale helpers (deterministic + SLM sanitizer) ───────────────────────
_APOLOGY = ("죄송", "sorry", "i cannot", "저는 항상", "도움과 칭찬", "as an ai", "언어 모델")
# Specific crime-scenario terms a small zero-shot backbone tends to hallucinate. If the SLM cites
# one of these but it does NOT appear in the transcript, the explanation is fabricated (e.g. "도청"
# on a teleshopping call) — drop it and keep only the deterministic, channel-grounded rationale.
_FABRICATION_MARKERS = ("도청", "감청", "해킹", "납치", "유괴", "유출", "명의도용", "협박",
                        "송금", "이체", "안전계좌", "검찰", "경찰", "금융감독원", "국세청",
                        "대출", "환급", "OTP", "비밀번호", "인증번호", "주민번호", "카드번호")


def _sanitize_sentence(raw: str, transcript: str = None):
    """Clean a small-LLM completion into at most one readable Korean sentence, or None.

    Also rejects sentences that invent a specific crime-scenario term (도청/이체/검찰/…) not present
    in the transcript — a zero-shot backbone hallucinating a scenario the call never contained."""
    if not raw:
        return None
    s = raw.strip().strip('"').strip()
    s = re.sub(r"^\s*(판정|이유|답변?|답)\s*[:：]?\s*", "", s)
    # first sentence only
    m = re.search(r"^(.+?[.。!?])", s, flags=re.S)
    if m:
        s = m.group(1)
    s = s.replace("\n", " ").strip()
    low = s.lower()
    if any(a in low for a in _APOLOGY):        # model refused / boilerplate
        return None
    hangul = sum(1 for c in s if "가" <= c <= "힣")
    if hangul < 4 or len(s) > 90:              # too little Korean / too long = unreliable
        return None
    if transcript is not None:                 # grounding: no invented scenario absent from the call
        tx = transcript or ""
        for term in _FABRICATION_MARKERS:
            if term in s and term not in tx:
                return None
    return s


def deterministic_rationale(diag: dict, verdict: str, transcript: str) -> str:
    """Always-readable, channel-grounded reason in English (no LLM needed). Any quoted cues stay in
    the transcript's own language (Korean call tokens)."""
    from miltl.native.explain import _scan_cues
    XM = float(diag.get("XM", 0.0)); E = float(diag.get("E", 0.0)); T = float(diag.get("T", 0.0))
    cues = _scan_cues(transcript or "")
    if verdict == "harm":
        base = ("warm wording over a low-valence (cold) voice — cross-modal contradiction" if XM >= 0.2
                else "a threatening / high-pressure vocal signal")
        extra = f", high vocal arousal E={E:.2f}" if E >= 0.6 else ""
        s = f"Harm rationale: observed {base} (XM={XM:.2f}{extra})."
        if cues:
            s += " Transcript cues — " + "; ".join(cues[:2]) + "."
        return s
    s = (f"Benign: cooperative, coherent dialogue with low cross-modal contradiction "
         f"(XM={XM:.2f}, naturalness/engagement T={T:.2f}).")
    if not cues:
        s += " No impersonation / transfer-inducement / threat vocabulary."
    return s


# ── ASR runtime (optional; faster-whisper) — for wave-only uploads ─────────
class ASRRuntime:
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model = None
        self._lock = threading.Lock()
        self.status = "not_loaded"
        self.device = "cpu"
        self.note = ""

    def available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure(self):
        with self._lock:
            if self._model is not None:
                return
            from faster_whisper import WhisperModel
            self.status = "loading"
            # Prefer the detected GPU so the streaming RTF actually drops on CUDA (the honest
            # "GPU/mobile keeps up" claim). But faster-whisper's CTranslate2 backend is a
            # SEPARATE build from torch: a CPU-only CTranslate2 wheel raises "not compiled with
            # CUDA support" even on a DGX. So try CUDA, and fall back to CPU int8 on any failure
            # (wrong wheel, no fp16, OOM). MPS has no CTranslate2 backend → CPU too.
            if _detect_runtime()["device"] == "cuda":
                try:
                    self._model = WhisperModel(self.model_size, device="cuda", compute_type="float16")
                    self.device = "cuda"
                    self.status = "ready"
                    return
                except Exception as e:  # noqa: BLE001
                    self.note = (f"faster-whisper CUDA unavailable ({type(e).__name__}: {e}); "
                                 "using CPU int8. For GPU ASR install a CUDA-enabled CTranslate2 "
                                 "(pip install --force-reinstall --no-binary :none: ctranslate2, "
                                 "or the matching CUDA wheel).")
                    print("[asr] " + self.note, flush=True)
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            self.device = "cpu"
            self.status = "ready"

    def load(self):
        """Force model construction now (resolves self.device / self.note before streaming)."""
        self._ensure()

    def transcribe(self, pcm: np.ndarray, sr: int = 16000) -> str:
        self._ensure()
        segments, _ = self._model.transcribe(pcm.astype(np.float32), language="ko",
                                             beam_size=1, vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()

    def transcribe_chunk(self, pcm: np.ndarray, sr: int = 16000) -> str:
        """Online per-nibble ASR: transcribe ONE ~8s chunk with no look-ahead. VAD is off
        (a hard VAD can nuke a short speech chunk); returns the chunk's words, possibly empty."""
        self._ensure()
        if pcm is None or len(pcm) < int(0.2 * sr):
            return ""
        segments, _ = self._model.transcribe(pcm.astype(np.float32), language="ko",
                                             beam_size=1, vad_filter=False, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()


# ── resource meters (stdlib, /proc) ─────────────────────────────────────────
_CLK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
_NCPU = os.cpu_count() or 1
_prev_cpu = {"t": None, "jiffies": None}


def _proc_metrics():
    rss_mb = 0.0
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                rss_mb = float(line.split()[1]) / 1024.0
                break
    except OSError:
        pass
    cpu_pct = None
    try:
        parts = open("/proc/self/stat").read().split()
        jiff = int(parts[13]) + int(parts[14])
        now = time.perf_counter()
        if _prev_cpu["t"] is not None:
            dt = now - _prev_cpu["t"]
            if dt > 0:
                cpu_pct = 100.0 * (jiff - _prev_cpu["jiffies"]) / _CLK / dt
        _prev_cpu["t"], _prev_cpu["jiffies"] = now, jiff
    except OSError:
        pass
    return {"rss_mb": round(rss_mb, 1),
            "cpu_pct": None if cpu_pct is None else round(min(cpu_pct, 100.0 * _NCPU), 1),
            "ncpu": _NCPU}


_GPU_CACHE = {"t": None, "val": None}


def _gpu_metrics():
    """Device-wide GPU utilization/memory (all processes) via pynvml, else nvidia-smi. None if no GPU.
    Cached ~1.5 s so the 2 s status poll never spawns back-to-back nvidia-smi calls."""
    now = time.perf_counter()
    if _GPU_CACHE["t"] is not None and (now - _GPU_CACHE["t"]) < 1.5:
        return _GPU_CACHE["val"]
    val = None
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        u = pynvml.nvmlDeviceGetUtilizationRates(h)
        m = pynvml.nvmlDeviceGetMemoryInfo(h)
        name = pynvml.nvmlDeviceGetName(h)
        val = {"util": int(u.gpu), "mem_mb": round(m.used / 1e6), "mem_total_mb": round(m.total / 1e6),
               "name": name.decode() if isinstance(name, bytes) else name}
        pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001
        try:
            import subprocess
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,name",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2).stdout.strip()
            if out:
                a = [x.strip() for x in out.splitlines()[0].split(",")]
                val = {"util": int(float(a[0])), "mem_mb": int(float(a[1])),
                       "mem_total_mb": int(float(a[2])), "name": a[3]}
        except Exception:  # noqa: BLE001
            val = None
    _GPU_CACHE.update({"t": now, "val": val})
    return val


def _warm_cues(text: str, maxn: int = 6):
    """Warm/reassuring words actually present in the transcript — the TEXT side of XM. These are the
    exact tokens that raise `warmth`; showing them makes 'warm words vs cold voice' concrete."""
    from miltl.native.channel_teacher import _WARM
    seen = []
    for w in _WARM:
        if w in (text or "") and w not in seen:
            seen.append(w)
    return seen[:maxn]


# ── Legacy reference detectors (live, for CUSTOM input) ─────────────────────
class LegacyPanel:
    """Live corpus-classification legacy detectors for user-supplied (non-bundled) input.

    Budget picks (CPU-free, weights already in this repo): lexical keyword-density proxy
    (always on) and the frozen KorCCViD CatBoost tree (if `catboost` is installed).
    Bundled canonical cases do NOT use this — they display the recorded benchmark
    verdicts of the full detector panel instead (more authoritative).
    """

    LEX_TAU = 2.0

    def __init__(self):
        from miltl.native.channel_teacher import _SCAM, _THREAT, _DIRECTIVE, _URGENCY
        self._kw = list(_SCAM) + list(_THREAT) + list(_DIRECTIVE) + list(_URGENCY)
        self._tree = None
        self._tree_status = "not_loaded"
        threading.Thread(target=self._load_tree, daemon=True).start()

    def _load_tree(self):
        try:
            import catboost  # noqa: F401
            from adapters.baselines.tree_ensemble import CatBoostDetector
            det = CatBoostDetector()
            det.load(str(REPO_ROOT / "artifacts/frozen/korccvid/tree"))
            det._frozen = True
            self._tree = det
            self._tree_status = "ready"
        except ImportError:
            self._tree_status = "pip install catboost to enable"
        except Exception as e:  # noqa: BLE001
            self._tree_status = f"{type(e).__name__}: {e}"

    def _density(self, text: str) -> float:
        nw = max(len(text.split()), 1)
        return sum(text.count(w) for w in self._kw) / nw * 100.0

    def run(self, transcript: str):
        rows = []
        t0 = time.perf_counter()
        dens = self._density(transcript)
        rows.append({"name": "lexical keyword density", "modality": "text (vocabulary)",
                     "score": round(min(dens / 8.0, 1.0), 3),
                     "verdict": "harm" if dens >= self.LEX_TAU else "benign",
                     "detail": f"density {dens:.2f}/100w · τ {self.LEX_TAU}",
                     "latency_ms": round((time.perf_counter() - t0) * 1e3, 1)})
        if self._tree is not None:
            import types as _t
            t0 = time.perf_counter()
            try:
                s = float(self._tree.score(_t.SimpleNamespace(transcript=transcript,
                                                              audio_uri=None, meta={})))
                rows.append({"name": "tree ensemble (CatBoost, frozen KorCCViD)",
                             "modality": "text (hashed word counts)", "score": round(s, 3),
                             "verdict": "harm" if s >= 0.5 else "benign", "detail": f"P={s:.3f}",
                             "latency_ms": round((time.perf_counter() - t0) * 1e3, 1)})
            except Exception as e:  # noqa: BLE001
                rows.append({"name": "tree ensemble (CatBoost, frozen KorCCViD)", "status": str(e)})
        else:
            rows.append({"name": "tree ensemble (CatBoost, frozen KorCCViD)", "status": self._tree_status})
        return rows


# ── bundled canonical cases ─────────────────────────────────────────────────
def _load_cases():
    if not CASES_FILE.is_file():
        return {"cases": [], "provenance": "", "sources": ""}
    return json.loads(CASES_FILE.read_text(encoding="utf-8"))


def _benign_texts_for_calib():
    """Benign transcripts for text-mode calibration: bundled synthetic benign (self-authored)."""
    synth = REPO_ROOT / "artifacts/rounds/canonical/synth_856444.jsonl"
    texts = []
    if synth.is_file():
        for line in synth.open(encoding="utf-8"):
            d = json.loads(line)
            if d["class"] == "benign":
                texts.append(d["transcript"])
            if len(texts) >= 40:
                break
    if not texts:
        texts = ["네 안녕하세요 문의 주셔서 감사합니다 천천히 말씀해 주세요 네 알겠습니다"] * 20
    return texts


# ── audio decode ────────────────────────────────────────────────────────────
def _decode_audio(raw: bytes, filename: str):
    """bytes → (pcm float32 [-1,1] mono 16k, sr, duration_s). wav via stdlib; mp3/mp4/m4a via
    ffmpeg (system, or the pip-bundled imageio-ffmpeg — no system install needed)."""
    suffix = Path(filename or "upload.wav").suffix or ".wav"
    from miltl.nibble.audio_decode import decode_to_pcm
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(raw)
        tmp = f.name
    try:
        pcm, sr = decode_to_pcm(tmp, sr=16000)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return pcm.astype(np.float32), sr, len(pcm) / float(sr or 16000)


# ── session orchestration (SSE streaming) ───────────────────────────────────
class Session:
    def __init__(self, engine: L1Engine, transcript: str, prosody: str, step_ms: int,
                 audio_raw: bytes | None = None, audio_name: str = "", asr: ASRRuntime | None = None,
                 live: bool = False):
        self.id = uuid.uuid4().hex[:12]
        self.engine = engine
        self.transcript = transcript
        self.prosody = prosody
        self.step_ms = max(0, min(step_ms, 3000))
        self.audio_raw = audio_raw
        self.audio_name = audio_name
        self.asr = asr
        self.live = live
        self.asr_used = False
        self.events: list = []
        self.done = threading.Event()
        self.final = None
        self.diag = None
        self.p2 = None
        threading.Thread(target=self._run, daemon=True).start()

    def _emit(self, ev):
        self.events.append(ev)

    def _run(self):
        try:
            self._run_inner()
        except Exception as e:  # noqa: BLE001
            self._emit({"type": "final", "band": "error", "note": f"{type(e).__name__}: {e}",
                        "sys": _proc_metrics()})
            self.done.set()

    def _run_inner(self):
        eng = self.engine
        pcm = None
        # 1) real audio path (upload) — decode + (optional) ASR
        if self.audio_raw:
            self._emit({"type": "prep", "msg": "decoding audio…", "sys": _proc_metrics()})
            pcm, sr, dur = _decode_audio(self.audio_raw, self.audio_name)
            self._emit({"type": "prep", "msg": f"decoded {dur:.1f}s @ {sr} Hz", "sys": _proc_metrics()})
            # Live online-streaming path: transcribe nibble-by-nibble as the call plays (no look-ahead).
            if self.live and not self.transcript.strip() and self.asr and self.asr.available():
                self._run_stream(pcm, sr, dur)
                return
            # Bound the decision to the observation envelope (same deadline as scoring / streaming):
            # a 37-minute upload is decided from its first DECISION_NIBBLES × 8 s, not transcribed whole.
            deadline_samp = int(min(DECISION_NIBBLES, MAX_NIBBLES) * NIBBLE_SEC * sr)
            if len(pcm) > deadline_samp:
                pcm = pcm[:deadline_samp]
                self._emit({"type": "prep", "msg": f"call {dur:.0f}s > decision envelope "
                            f"{min(DECISION_NIBBLES, MAX_NIBBLES)*NIBBLE_SEC:.0f}s → deciding on the "
                            f"first {min(DECISION_NIBBLES, MAX_NIBBLES)*NIBBLE_SEC:.0f}s", "sys": _proc_metrics()})
            if not self.transcript.strip():
                if self.asr and self.asr.available():
                    self._emit({"type": "prep", "msg": "transcribing (faster-whisper, CPU)…",
                                "sys": _proc_metrics()})
                    t0 = time.perf_counter()
                    self.transcript = self.asr.transcribe(pcm, sr)
                    self.asr_used = True
                    self._emit({"type": "prep",
                                "msg": f"ASR done ({time.perf_counter()-t0:.1f}s): "
                                       f"{self.transcript[:80]}…", "sys": _proc_metrics()})
                else:
                    self.final = {"band": "error",
                                  "note": "wave uploaded without transcript, but faster-whisper is not "
                                          "installed — `pip install faster-whisper`, or paste a transcript."}
                    self._emit({"type": "final", **self.final, "sys": _proc_metrics()})
                    self.done.set()
                    return
            nci, has_audio, tf = eng.featurize_real_audio(self.transcript, pcm)
        else:
            nci, has_audio, tf = eng.featurize(self.transcript, self.prosody)

        ok, n_words, anchor = eng.anchor_ok(self.transcript, has_audio)
        self._emit({"type": "start", "n_nibbles": int(nci.n_valid), "n_words": n_words,
                    "anchor": anchor, "anchor_ok": ok, "has_audio": has_audio,
                    "prosody": tf.get("prosody_kind"), "prosody_real": tf.get("prosody_real"),
                    "transcript": self.transcript, "asr": self.asr_used,
                    "timings": tf, "sys": _proc_metrics()})
        if not ok:
            self.final = {"band": "undecidable", "p1": 0.0,
                          "note": f"below observation anchor ({n_words} < {anchor} words) → 0.0 (undecidable)"}
            self._emit({"type": "final", **self.final, "sys": _proc_metrics()})
            self.done.set()
            return
        step = None
        for k in range(1, int(nci.n_valid) + 1):
            step = eng.score_prefix(nci, has_audio, k)
            step["type"] = "step"
            step["warm_cues"] = _warm_cues(self.transcript)
            step["sys"] = _proc_metrics()
            self._emit(step)
            if self.step_ms:
                time.sleep(self.step_ms / 1e3)
        diag = dict(step["mean"])
        diag["audio"] = 1 if has_audio else 0
        self.diag = diag
        p1, band = step["p1"], step["band"]
        decision = {"benign": "benign", "harm": "harm", "escalate": "escalate→L2"}[band]
        xai = explain_decision(diag, "harm" if band == "harm" else ("benign" if band == "benign" else "escalate"),
                               self.transcript, p1=p1)
        self.final = {"band": band, "decision": decision, "p1": p1, "risk": step["risk"],
                      "diag": {k2: round(float(v), 4) for k2, v in diag.items()},
                      "xai": xai, "l1_ms_last": step["l1_ms"]}
        self._emit({"type": "final", **self.final, "sys": _proc_metrics()})
        self.done.set()

    def _run_stream(self, pcm, sr, dur):
        """True online streaming: process one 8 s nibble at a time as it arrives, transcribing
        with no look-ahead and re-scoring the growing prefix. NOTHING is throttled — each chunk
        is processed as fast as the hardware allows, so the emitted per-chunk wall latency is the
        honest streaming cost. The client compares the processing frontier against real playback
        to show the buffer (positive = ahead, negative = falling behind) and the real-time factor."""
        eng = self.engine
        self.asr_used = True
        # Resolve the ASR device up front (builds the model) so the UI shows the true device and
        # any CUDA→CPU fallback note before the first nibble streams.
        self._emit({"type": "prep", "msg": "loading streaming ASR…", "sys": _proc_metrics()})
        try:
            self.asr.load()
        except Exception as e:  # noqa: BLE001
            self._emit({"type": "final", "band": "error", "asr": True,
                        "note": f"ASR load failed: {type(e).__name__}: {e}", "sys": _proc_metrics()})
            self.done.set()
            return
        if getattr(self.asr, "note", ""):
            self._emit({"type": "prep", "msg": self.asr.note, "sys": _proc_metrics()})
        nib = max(1, int(round(NIBBLE_SEC * sr)))
        n_total = max(1, int(np.ceil(len(pcm) / nib)))          # full length of the uploaded call
        deadline_n = max(1, min(DECISION_NIBBLES, MAX_NIBBLES))  # decision envelope (cannot score past MAX_NIBBLES)
        n_chunks = min(n_total, deadline_n)                     # stop streaming ASR at the deadline
        deadline_s = deadline_n * NIBBLE_SEC
        truncated = n_total > deadline_n
        self._emit({"type": "start", "live": True, "n_nibbles": n_chunks, "n_words": 0,
                    "anchor": ANCHOR_AUDIO, "anchor_ok": True, "has_audio": True,
                    "prosody": "real-audio", "prosody_real": True, "transcript": "",
                    "asr": True, "duration_s": round(dur, 2), "nibble_s": NIBBLE_SEC,
                    "deadline_nibbles": deadline_n, "deadline_s": round(deadline_s, 1),
                    "total_nibbles": n_total, "decided_nibbles": n_chunks, "truncated": truncated,
                    "asr_device": getattr(self.asr, "device", "cpu"),
                    "asr_note": getattr(self.asr, "note", ""),
                    "timings": {"featurize_ms": 0.0}, "sys": _proc_metrics()})
        words: list = []
        wall0 = time.perf_counter()
        asr_ms_tot = miltl_ms_tot = 0.0
        budget_ms = NIBBLE_SEC * 1e3
        step = None
        for k in range(1, n_chunks + 1):
            # Real-time pacing: a live call arrives at 1× — nibble k's 8 s of audio is complete only
            # at real-time k×8 s, so we WAIT for it before transcribing (you cannot transcribe audio
            # that has not been spoken yet). This keeps the processing frontier from ever running
            # ahead of actual playback. The edge-practicality question is not "how fast can it batch a
            # file" but "does each nibble finish inside its 8 s budget"; the wait is excluded from the
            # measured compute time (asr_ms/miltl_ms), so the RTF still reflects true compute headroom.
            arrival = k * NIBBLE_SEC
            wait = arrival - (time.perf_counter() - wall0)
            if wait > 0:
                time.sleep(wait)
            chunk = pcm[(k - 1) * nib: k * nib]
            t = time.perf_counter()
            txt = self.asr.transcribe_chunk(chunk, sr)          # online ASR, this 8 s only
            asr_ms = (time.perf_counter() - t) * 1e3
            asr_ms_tot += asr_ms
            if txt:
                words.extend(txt.split())
            transcript_prefix = " ".join(words)
            self.transcript = transcript_prefix
            t = time.perf_counter()                             # MiLTL compute on the prefix so far
            nci, has_audio, tf = eng.featurize_real_audio(transcript_prefix or " ", pcm[: k * nib])
            kk = max(1, int(nci.n_valid))
            step = eng.score_prefix(nci, True, kk)
            miltl_ms = (time.perf_counter() - t) * 1e3
            miltl_ms_tot += miltl_ms
            nibble_ms = asr_ms + miltl_ms                       # compute for THIS nibble (vs 8 s budget)
            behind_s = max(0.0, (time.perf_counter() - wall0) - arrival)  # real-time backlog (0 = keeping up)
            audio_t = min(k * NIBBLE_SEC, dur)
            n_words_now = len(words)
            decidable = n_words_now >= ANCHOR_AUDIO             # enough observation → Gate-1 verdict is live
            step_xai = None
            if decidable:                                       # lock a running L1 verdict (updates each nibble)
                d = dict(step["mean"]); d["audio"] = 1
                self.diag = d                                   # enables mid-stream L2 scoring
                _vb = step["band"] if step["band"] in ("harm", "benign") else "escalate"
                step_xai = explain_decision(d, _vb, transcript_prefix, p1=step["p1"])
            self._emit({"type": "step", "k": k, "audio_time": round(audio_t, 2), "xai": step_xai,
                        "chunk_text": txt, "transcript": transcript_prefix,
                        "asr_ms": round(asr_ms, 1), "miltl_ms": round(miltl_ms, 2),
                        "asr_ms_tot": round(asr_ms_tot, 1), "miltl_ms_tot": round(miltl_ms_tot, 2),
                        "nibble_ms": round(nibble_ms, 1), "budget_ms": round(budget_ms, 0),
                        "over_budget": bool(nibble_ms > budget_ms), "behind_s": round(behind_s, 2),
                        "wall_ms": round((time.perf_counter() - wall0) * 1e3, 1),
                        "decidable": decidable, "n_words": n_words_now, "anchor": ANCHOR_AUDIO,
                        "nibble": step["nibble"], "decomp": step["decomp"],
                        "warm_cues": _warm_cues(transcript_prefix),
                        "mean": step["mean"], "risk": step["risk"],
                        "p1": step["p1"], "band": step["band"], "l1_ms": step["l1_ms"],
                        "sys": _proc_metrics()})
        # final verdict from the full-call prefix
        n_words = len(words)
        ok = n_words >= ANCHOR_AUDIO
        if step is None or not ok:
            self.final = {"band": "undecidable", "p1": 0.0,
                          "note": f"below observation anchor ({n_words} < {ANCHOR_AUDIO} words) → undecidable"}
            self._emit({"type": "final", **self.final, "asr": True,
                        "n_words": n_words, "sys": _proc_metrics()})
            self.done.set()
            return
        diag = dict(step["mean"]); diag["audio"] = 1
        self.diag = diag
        p1, band = step["p1"], step["band"]
        decision = {"benign": "benign", "harm": "harm", "escalate": "escalate→L2"}[band]
        xai = explain_decision(diag, "harm" if band == "harm" else ("benign" if band == "benign" else "escalate"),
                               self.transcript, p1=p1)
        wall_s = time.perf_counter() - wall0
        proc_s = min(n_chunks * NIBBLE_SEC, dur) or dur
        compute_s = (asr_ms_tot + miltl_ms_tot) / 1e3          # actual compute (excludes real-time waiting)
        self.final = {"band": band, "decision": decision, "p1": p1, "risk": step["risk"],
                      "diag": {k2: round(float(v), 4) for k2, v in diag.items()},
                      "xai": xai, "l1_ms_last": step["l1_ms"], "asr": True, "n_words": n_words,
                      "asr_rtf": round((asr_ms_tot / 1e3) / proc_s, 3) if proc_s else None,
                      "miltl_rtf": round((miltl_ms_tot / 1e3) / proc_s, 4) if proc_s else None,
                      "compute_s": round(compute_s, 2), "wall_s": round(wall_s, 2), "audio_s": round(dur, 2),
                      "decided_s": round(n_chunks * NIBBLE_SEC, 1), "truncated": truncated,
                      "total_nibbles": n_total, "decided_nibbles": n_chunks}
        self._emit({"type": "final", **self.final, "sys": _proc_metrics()})
        self.done.set()


SESSIONS: dict = {}


# ── HTTP server ─────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    engine: L1Engine = None
    l2: L2Runtime = None
    legacy: LegacyPanel = None
    asr: ASRRuntime = None
    casesdoc: dict = None

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            body = PAGE_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif u.path == "/api/cases":
            self._json({**self.casesdoc,
                        "calib": self.engine.calib_src,
                        "asr": ("available" if (self.asr and self.asr.available()) else "not installed"),
                        "op": {"tau_low": TAU_LOW, "tau_high": TAU_HIGH,
                               "rule": "risk = E − 2·T + I + XM (+2·(F−0.5) if no audio)",
                               "anchor_audio": ANCHOR_AUDIO, "anchor_text": ANCHOR_TEXT}})
        elif u.path == "/api/status":
            self._json({"l2": {"status": self.l2.status if self.l2 else "disabled",
                               "detail": self.l2.detail if self.l2 else "--no-gate2",
                               "model": self.l2.model_name if self.l2 else None,
                               "device": self.l2.device if self.l2 else None,
                               "adapter": bool(self.l2.adapter_on) if self.l2 else False,
                               "load_s": self.l2.load_s if self.l2 else None},
                        "runtime": _detect_runtime(),
                        "gpu": _gpu_metrics(),
                        "sys": _proc_metrics()})
        elif u.path == "/api/stream":
            sid = parse_qs(u.query).get("sid", [""])[0]
            s = SESSIONS.get(sid)
            if not s:
                self._json({"error": "unknown session"}, 404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            i = 0
            try:
                while True:
                    while i < len(s.events):
                        ev = s.events[i]
                        i += 1
                        self.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode())
                        self.wfile.flush()
                        if ev.get("type") == "final":
                            return
                    if s.done.is_set() and i >= len(s.events):
                        return
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                return
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        if u.path == "/api/run":
            transcript = (req.get("transcript") or "").strip()
            audio_raw = None
            audio_name = req.get("audio_name", "")
            if req.get("audio_b64"):
                try:
                    audio_raw = base64.b64decode(req["audio_b64"])
                except Exception:  # noqa: BLE001
                    self._json({"error": "bad audio encoding"}, 400)
                    return
            if not transcript and not audio_raw:
                self._json({"error": "provide a transcript or an audio file"}, 400)
                return
            s = Session(self.engine, transcript, req.get("prosody", "none"),
                        int(req.get("step_ms", 250)), audio_raw=audio_raw,
                        audio_name=audio_name, asr=self.asr,
                        live=bool(req.get("stream") == "live"))
            SESSIONS[s.id] = s
            self._json({"sid": s.id})
        elif u.path == "/api/legacy":
            s = SESSIONS.get(req.get("sid", ""))
            if not s or not s.done.is_set():
                self._json({"error": "session not finished"}, 400)
                return
            self._json({"rows": self.legacy.run(s.transcript),
                        "note": "LIVE demo detectors on your input (lexical proxy + frozen tree). "
                                "For bundled cases the recorded canonical verdicts are shown instead.",
                        "sys": _proc_metrics()})
        elif u.path == "/api/l2/load":
            if not self.l2:
                self._json({"error": "gate-2 disabled (--no-gate2)"}, 400)
                return
            self.l2.start_loading()
            self._json({"status": self.l2.status})
        elif u.path in ("/api/l2/score", "/api/l2/rationale"):
            if not self.l2 or self.l2.status != "ready":
                self._json({"error": f"gate-2 not ready ({self.l2.status if self.l2 else 'disabled'})"}, 400)
                return
            s = SESSIONS.get(req.get("sid", ""))
            if not s or s.diag is None:                          # decidable (anchor met) — no need to wait for loop end
                self._json({"error": "not decidable yet (waiting for the observation anchor)"}, 400)
                return
            try:
                if u.path == "/api/l2/score":
                    p2, dt = self.l2.score(s.transcript, s.diag)
                    s.p2 = p2
                    final = TAU_LOW + p2 * (TAU_HIGH - TAU_LOW)
                    self._json({"p2": round(p2, 4), "final": round(final, 4),
                                "mapping": f"final = {TAU_LOW} + P(harm)·({TAU_HIGH}−{TAU_LOW})",
                                "latency_s": round(dt, 2), "sys": _proc_metrics()})
                else:
                    verdict = "harm" if (s.p2 is not None and s.p2 >= 0.5) else "benign"
                    out = self.l2.concise_rationale(s.transcript, s.diag, verdict)
                    self._json({"reason": out["reason"], "slm": out["slm"], "verdict": verdict,
                                "model": self.l2.model_name, "adapter": bool(self.l2.adapter_on),
                                "latency_s": out["latency_s"], "sys": _proc_metrics()})
            except Exception as e:  # noqa: BLE001
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)
        else:
            self._json({"error": "not found"}, 404)


# ── selftest ────────────────────────────────────────────────────────────────
def selftest() -> int:
    eng = L1Engine()
    print(f"[selftest] calib: {eng.calib_src} (fit {eng.calib_fit_s:.2f}s)")
    doc = _load_cases()
    cases = doc.get("cases", [])
    print(f"[selftest] {len(cases)} bundled canonical cases (AI-Hub excluded: "
          f"{'none' if not any('emotion' in c['cid'] or 'aihub' in c['cid'] for c in cases) else 'LEAK!'})")
    import statistics
    lat = []
    for c in cases:
        nci, has_audio, _ = eng.featurize(c["transcript"], c["prosody"])
        ok, n, anchor = eng.anchor_ok(c["transcript"], has_audio)
        if not ok:
            print(f"  {c['cid']:<16} gated ({n}<{anchor})")
            continue
        t0 = time.perf_counter()
        step = eng.score_prefix(nci, has_audio, int(nci.n_valid))
        lat.append((time.perf_counter() - t0) * 1e3)
        v = c["verdicts"]
        canon = " ".join(f"{k.split('(')[0].split('-')[0]}={v[k]['outcome']}"
                         for k in ["MiLTL-Cascade", "hf-encoder", "tree", "Bllossom-B3", "Wave-Seq(audio-only)"] if k in v)
        print(f"  {c['cid']:<16} [{c['group'][:34]:<34}] demo-p1={step['p1']:.2f}({step['band']}) "
              f"XM={step['mean']['XM']:.2f} | canonical: {canon}")
    if lat:
        print(f"[selftest] demo L1 latency median={statistics.median(lat):.2f} ms")
    m = _proc_metrics()
    print(f"[selftest] RSS={m['rss_mb']} MB · ncpu={m['ncpu']}")
    print("[selftest] OK — bundled cases carry verbatim canonical verdicts; demo L1 re-runs the mechanism.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="MiLTL standalone CPU demo (web UI)")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--gate2-model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="Gate-2 backbone (canonical: Qwen/Qwen2.5-1.5B-Instruct)")
    ap.add_argument("--no-gate2", action="store_true", help="L1-only demo (pure numpy)")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads for Gate-2 (0=auto)")
    ap.add_argument("--calib", default=None, help="path to a released Calib JSON (auto-detects "
                    "artifacts/models/calib.release.json or channel_extractors.pt)")
    ap.add_argument("--gate2-adapter", default=None, help="path to the release Gate-2 LoRA SFT "
                    "adapter dir (auto-detects artifacts/models/gate2_adapter_1.5b/)")
    ap.add_argument("--asr-model", default="base", help="faster-whisper size for wave-only uploads")
    ap.add_argument("--max-nibbles", type=int, default=MAX_NIBBLES,
                    help=f"decision deadline in 8 s nibbles (canonical {MAX_NIBBLES} = "
                         f"{MAX_NIBBLES*NIBBLE_SEC:.0f}s; capped at {MAX_NIBBLES} — scoring cannot see past it)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    global DECISION_NIBBLES
    DECISION_NIBBLES = max(1, min(int(args.max_nibbles), MAX_NIBBLES))

    if args.selftest:
        raise SystemExit(selftest())

    # Resolve the Gate-2 SFT adapter: explicit flag, else auto-detect the release dir. Verify the
    # adapter's base model matches the chosen backbone (the release adapter is for 1.5B) so an
    # accidental 0.5B run doesn't crash — otherwise skip it with a clear hint.
    adapter = args.gate2_adapter
    if adapter is None and _GATE2_ADAPTER_DIR.exists():
        adapter = str(_GATE2_ADAPTER_DIR)
    if adapter and not (Path(adapter) / "adapter_config.json").exists():
        print(f"[gate2] {adapter}/adapter_config.json is missing — PEFT cannot attach the adapter "
              f"(upload it next to adapter_model.safetensors). Running zero-shot base for now.")
        adapter = None
    if adapter:
        try:
            base = json.loads((Path(adapter) / "adapter_config.json").read_text()).get("base_model_name_or_path", "")
        except Exception:  # noqa: BLE001
            base = ""
        if base and base.split("/")[-1] not in args.gate2_model:
            print(f"[gate2] adapter is for '{base}' but backbone is '{args.gate2_model}' — "
                  f"run with --gate2-model {base} to use it; ignoring adapter for now.")
            adapter = None

    Handler.engine = L1Engine(calib_path=args.calib)
    Handler.legacy = LegacyPanel()
    Handler.asr = ASRRuntime(args.asr_model)
    Handler.casesdoc = _load_cases()
    Handler.l2 = None if args.no_gate2 else L2Runtime(args.gate2_model, threads=args.threads, adapter_path=adapter)
    print(f"[demo] L1 ready — {Handler.engine.calib_src}")
    print(f"[demo] cases: {len(Handler.casesdoc.get('cases', []))} canonical · "
          f"ASR: {'available' if Handler.asr.available() else 'not installed (wave-only needs faster-whisper)'}")
    print(f"[demo] Gate-2: {'disabled' if args.no_gate2 else args.gate2_model + (' + SFT adapter' if adapter else ' (zero-shot base)') + ' (loads on demand)'}")
    print(f"[demo] open  http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


# ── web page ────────────────────────────────────────────────────────────────
PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MiLTL demo — Multimodal In-call Lightweight Threat Locator</title>
<style>
:root { --bg:#0e1116; --panel:#161b24; --line:#2a3140; --txt:#dfe6f1; --dim:#8b96a8;
        --T:#4cc9f0; --I:#f4a261; --F:#e76f51; --E:#e9c46a; --X:#c77dff;
        --benign:#2fbf71; --esc:#f4a261; --harm:#ef476f; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--txt); font:14px/1.5 "Segoe UI",system-ui,sans-serif; }
header { padding:14px 22px; border-bottom:1px solid var(--line); }
header h1 { margin:0; font-size:17px; } header .sub { color:var(--dim); font-size:12px; }
main { display:grid; grid-template-columns: 360px 1fr 330px; gap:14px; padding:14px 22px; }
@media (max-width:1150px){ main{ grid-template-columns:1fr; } }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; }
.panel h2 { margin:0 0 10px; font-size:13px; text-transform:uppercase; letter-spacing:.06em; color:var(--dim); }
select,textarea,button,input[type=range],input[type=file] { width:100%; background:#0b0e13; color:var(--txt);
  border:1px solid var(--line); border-radius:7px; padding:8px; font:inherit; }
textarea { height:120px; resize:vertical; }
button { cursor:pointer; background:#20324e; border-color:#31507e; margin-top:8px; }
button:hover { background:#294067; } button:disabled { opacity:.45; cursor:default; }
.row { display:flex; gap:8px; } .row > * { flex:1; }
.lamp { display:inline-block; padding:3px 12px; border-radius:20px; font-weight:600; font-size:12px;
        background:#252c39; color:var(--dim); border:1px solid var(--line); }
.lamp.on-benign { background:var(--benign); color:#06130c; border-color:var(--benign);}
.lamp.on-esc    { background:var(--esc);   color:#211302; border-color:var(--esc);}
.lamp.on-harm   { background:var(--harm);  color:#210409; border-color:var(--harm);}
.lamp.on-idle   { background:#252c39; }
.lamp.pulse { animation:pu 1s infinite; } @keyframes pu { 50%{ filter:brightness(1.4);} }
canvas { width:100%; background:#0b0e13; border:1px solid var(--line); border-radius:7px; }
.kv { display:grid; grid-template-columns:auto 1fr; gap:2px 12px; font-size:12.5px; }
.kv b { color:var(--dim); font-weight:500; }
.big { font-size:26px; font-weight:700; }
.bar { height:9px; background:#0b0e13; border-radius:6px; overflow:hidden; border:1px solid var(--line); }
.bar i { display:block; height:100%; width:0%; transition:width .2s; }
.legend span { display:inline-block; margin-right:10px; font-size:12px; }
.dot { display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:4px; vertical-align:-1px;}
.note { font-size:11.5px; color:var(--dim); margin-top:8px; }
pre { white-space:pre-wrap; font-size:12px; background:#0b0e13; border:1px solid var(--line);
      border-radius:7px; padding:9px; max-height:200px; overflow:auto; }
.gauge { text-align:center; margin:6px 0; }
hr { border:0; border-top:1px solid var(--line); margin:12px 0; }
.small { font-size:12px; color:var(--dim); }
.tbl { width:100%; border-collapse:collapse; font-size:12px; }
.tbl td { padding:4px 6px; border-bottom:1px solid var(--line); }
.tag { font-size:10.5px; padding:1px 6px; border-radius:10px; font-weight:600; }
.tag.ok { background:rgba(47,191,113,.18); color:var(--benign); }
.tag.no { background:rgba(239,71,111,.18); color:var(--harm); }
.why { font-size:12px; color:#c7d0de; background:#10151d; border-left:3px solid var(--X);
       border-radius:6px; padding:8px 10px; margin:8px 0; }
.miltl-row { background:rgba(199,125,255,.08); }
</style></head><body>
<header>
  <h1>MiLTL — Multimodal In-call Lightweight Threat Locator · CPU demo</h1>
  <div class="sub">Gate-1 analytic channels (pure numpy) → banding τ 0.40 / 0.90 → Gate-2 SLM (CPU, on-demand).
  Bundled cases quote the 5-seed canonical benchmark verbatim (transcripts + recorded per-detector verdicts).</div>
  <div class="sub" style="color:var(--esc);margin-top:4px">⚠️ <b>Korean-only / 한국어 전용.</b>
  The channel lexicons, calibration, Gate-2 prompts, and auto-ASR (language=ko) are all Korean-based —
  English or other-language transcripts/audio will not be judged correctly. 영어 등 다른 언어 입력은 정상 작동하지 않습니다.</div>
</header>
<main>
  <section class="panel">
    <h2>Input</h2>
    <select id="case"></select>
    <div class="why" id="why" style="display:none"></div>
    <textarea id="transcript" placeholder="…or paste any Korean call transcript here (Korean only)"></textarea>
    <div class="note" id="asrbadge" style="display:none;border-left:3px solid var(--benign);padding-left:8px">
      🎙 <b>Auto-transcribed (faster-whisper).</b> The text above is what the demo transcribed from your
      audio — this exact text is what the lexical keyword-density proxy and the CatBoost tree ensemble judge.
    </div>
    <label class="small">Upload audio (.wav / .mp3 / .mp4 / .m4a) — real prosody; auto-ASR if no transcript</label>
    <input type="file" id="audio" accept=".wav,.mp3,.mp4,.m4a,.ogg,.flac">
    <audio id="player" controls style="width:100%;margin-top:6px;display:none"></audio>
    <label class="small" id="livewrap" style="display:none;align-items:center;gap:6px;margin-top:4px">
      <input type="checkbox" id="livemode" checked> Live streaming — play the call and transcribe /
      score each 8 s nibble online (no look-ahead) as it plays</label>
    <div class="row">
      <div><label class="small">Prosody (simulated; ignored if audio uploaded)</label>
        <select id="prosody">
          <option value="none">none (transcript-only)</option>
          <option value="warm">warm benign voice</option>
          <option value="cold">cold flat voice</option>
          <option value="cold-pressure">cold + pressure</option>
        </select></div>
      <div><label class="small">Nibble interval <span id="spv">250</span> ms</label>
        <input type="range" id="speed" min="0" max="1000" step="50" value="250"></div>
    </div>
    <button id="run">▶ Run call — live playback when audio + ASR, else stream nibbles</button>
    <div class="note" id="opnote"></div>
    <hr>
    <h2>Gate-2 (L2) runtime</h2>
    <div class="kv"><b>runtime</b><span id="runtime">detecting…</span><b>status</b><span id="l2status">…</span><b>model</b><span id="l2model">…</span></div>
    <label class="small" style="display:flex;align-items:center;gap:6px;margin:4px 0">
      <input type="checkbox" id="l2enable"> Enable Gate-2 (L2) — auto-on when a GPU is detected
    </label>
    <div class="note" id="l2warn" style="display:none;border-left:3px solid var(--esc);padding-left:8px">
      ⚠ <b>CPU-only device.</b> A 0.5B backbone reasons poorly on CPU and tends to flip a
      <i>benign</i> call to <i>harm</i>. Trust the deterministic Gate-1 band here; for a real
      L2 verdict use a GPU device or the canonical <code>--gate2-model Qwen/Qwen2.5-1.5B-Instruct</code>.
    </div>
    <button id="l2load">Load Gate-2 model (CPU)</button>
    <button id="l2score" disabled>⚖ L2 score this call (1 forward pass)</button>
    <button id="l2rat" disabled>🗒 Generate XAI rationale</button>
    <div class="note">L2 scoring = single forward pass (yes/no token log-probs) — no generation, which is what makes it laptop/mobile-feasible. The buttons stay disabled until L2 is enabled, loaded, and Gate-1 has landed a call in the escalate band.</div>
  </section>

  <section class="panel">
    <h2>Live channel trace — [L,5] per 8-second nibble <span class="small" style="color:var(--dim)">(y = channel activation 0–1; x = nibble)</span></h2>
    <div class="legend">
      <span><i class="dot" style="background:var(--T)"></i>T truth</span>
      <span><i class="dot" style="background:var(--I)"></i>I indeterminacy</span>
      <span><i class="dot" style="background:var(--F)"></i>F coercion</span>
      <span><i class="dot" style="background:var(--E)"></i>E arousal</span>
      <span><i class="dot" style="background:var(--X)"></i>XM cross-modal</span>
    </div>
    <details class="note" style="margin:4px 0">
      <summary style="cursor:pointer">How each channel is computed — all analytic, no black box</summary>
      Gate-1 has <b>0 learned weights</b> (Calib = 42 scalars); every channel is a named formula, not a
      RoBERTa embedding:<br>
      • <b>E arousal</b> = vocal energy + pitch + speaking rate (prosody)<br>
      • <b>F coercion</b> = 0.40·cold-voice + 0.30·subversion + 0.20·threat + 0.10·directive<br>
      • <b>I latent</b> = 0.85·XM + 0.15·(warmth × ask)<br>
      • <b>T truth</b> = coherence + balanced dominance + warmth − coercion<br>
      • <b>XM</b> = clip(warmth − voice-valence, 0, 1) × (0.5 + 0.5·dominance)<br>
      Prosody terms come from the audio; threat/directive/warmth from the transcript lexicon — so any
      verdict is fully auditable.
    </details>
    <canvas id="chart" height="200"></canvas>
    <h2 style="margin-top:14px">Risk trajectory — risk = E − 2·T + I + XM → p1</h2>
    <canvas id="riskchart" height="120"></canvas>
    <div class="row" style="margin-top:12px">
      <div class="gauge"><div class="small">Gate-1 p1 (demo)</div><div class="big" id="p1">–</div>
        <div class="bar"><i id="p1bar" style="background:var(--esc)"></i></div></div>
      <div class="gauge"><div class="small">band</div><div style="margin-top:8px">
        <span class="lamp" id="lampB">BENIGN</span>
        <span class="lamp" id="lampE">ESCALATE</span>
        <span class="lamp" id="lampH">HARM</span></div></div>
      <div class="gauge"><div class="small">L2 final</div><div class="big" id="finalv">–</div>
        <div class="small" id="finalmap"></div></div>
    </div>
    <hr>
    <h2>XM — cross-modal contradiction <span class="small" style="color:var(--dim)">MiLTL's unique channel: warm words vs low-valence (cold) voice</span></h2>
    <canvas id="xmchart" height="90"></canvas>
    <div class="row" style="margin-top:8px">
      <div class="gauge"><div class="small">🗣 Lexical warmth (text)</div>
        <div class="bar"><i id="xmw" style="background:var(--benign)"></i></div><div class="small" id="xmwv">–</div></div>
      <div class="gauge"><div class="small">🎙 Vocal valence V (voice)</div>
        <div class="bar"><i id="xmv" style="background:var(--T)"></i></div><div class="small" id="xmvv">–</div></div>
      <div class="gauge"><div class="small">XM = warmth − valence, ×dominance</div><div class="big" id="xmval">–</div></div>
    </div>
    <div class="note" id="xmcue" style="display:none"></div>
    <div class="note">XM fires when the <b>words are warm but the voice is cold</b> (scaled by vocal dominance)
      — a mismatch only a cross-modal model can see. A <b>text-only</b> LLM (Bllossom) and an
      <b>audio-only</b> model (Wave-Seq) each see one side and miss it; XM is what catches
      vocabulary-free grooming — exactly the benchmark false-negatives MiLTL recovers.</div>
    <hr>
    <h2>Interpretation (XAI)</h2>
    <pre id="xai">run a call to see the channel-grounded explanation…</pre>
    <h2>Gate-2 rationale</h2>
    <pre id="rat">not generated (optional; base Instruct model until the LoRA release)</pre>
  </section>

  <section class="panel">
    <h2>Gate activity</h2>
    <div class="kv">
      <b>L1 Gate-1</b><span><span class="lamp" id="l1lamp">idle</span></span>
      <b>L2 Gate-2</b><span><span class="lamp" id="l2lamp">idle</span></span>
      <b>nibbles</b><span id="nnib">–</span>
      <b>anchor</b><span id="anchor">–</span>
      <b>prosody</b><span id="hasaudio">–</span>
    </div>
    <div id="prep" class="note"></div>
    <div id="livebox" style="display:none">
      <hr>
      <h2>Live streaming (this device)</h2>
      <div class="kv">
        <b>▶ playback</b><span id="cl_play">0.0 s</span>
        <b>⚙ processed</b><span id="cl_proc">0.0 s</span>
        <b>buffer</b><span id="cl_lag">–</span>
        <b>⏱ decision deadline</b><span id="cl_deadline">–</span>
        <b>per-nibble compute</b><span id="cl_budget">–</span>
        <b>keeping up?</b><span id="cl_keepup">–</span>
      </div>
      <div class="bar" style="margin-top:6px"><i id="lagbar" style="background:var(--benign)"></i></div>
      <div class="note" id="deadlinenote" style="display:none;border-left:3px solid var(--esc);padding-left:8px"></div>
      <div class="kv" style="margin-top:6px">
        <b>ASR RTF</b><span id="rtf_asr">–</span>
        <b>MiLTL RTF</b><span id="rtf_miltl">–</span>
        <b>ASR device</b><span id="asrdev">–</span>
      </div>
      <div class="note" id="livenote">The call plays in real time (1×). Edge test: does each nibble finish
        inside its <b>8 s budget</b>? — green keeps up, red falls behind. <b>RTF</b> = compute ÷ audio (lower
        is better); the ASR front-end dominates it, MiLTL's own cascade stays far under 1× everywhere.</div>
    </div>
    <hr>
    <h2>Latency (this machine)</h2>
    <div class="kv">
      <b>featurize</b><span id="t_feat">–</span>
      <b>L1 / step</b><span id="t_l1">–</span>
      <b>L1 full call</b><span id="t_l1full">–</span>
      <b>L2 score</b><span id="t_l2">–</span>
      <b>L2 rationale</b><span id="t_rat">–</span>
    </div>
    <hr>
    <h2>Resources (this process)</h2>
    <div class="kv"><b>CPU</b><span id="cpu">–</span><b>RSS</b><span id="rss">–</span><b>cores</b><span id="ncpu">–</span></div>
    <div class="bar" style="margin-top:6px"><i id="cpubar" style="background:var(--T)"></i></div>
    <div class="bar" style="margin-top:6px"><i id="rssbar" style="background:var(--I)"></i></div>
    <div class="kv" style="margin-top:8px"><b>GPU (device-wide)</b><span id="gpu">–</span></div>
    <div class="bar" style="margin-top:6px"><i id="gpubar" style="background:var(--X)"></i></div>
    <div class="note">CPU = busy across all cores on a single 0–100 % scale. GPU util/memory are
      device-wide (all processes). RSS bar scale: 8 GB (mobile-class budget).</div>
    <hr>
    <h2 id="legtitle">Detector verdicts (same input)</h2>
    <div id="legend2" class="small">run a call — bundled cases show the recorded canonical verdicts of the full detector panel.</div>
    <div class="note" id="legnote"></div>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
let CASES = [], SID = null, steps = [], nTotal = 0, CUR = null, AUDIO_B64 = null, AUDIO_NAME = "", ASR_FILLED = false;
let AUDIO_FILE = null, PLAY_URL = null, ASR_OK = false, LIVE_ON = false, FRONTIER = 0, RAF = null, DUR = 0, DEADLINE_S = 0, ES = null, L2_DONE = false, DECIDED = false;
// Abort any in-flight run: close its SSE stream (else its late step events keep overwriting the
// transcript box — a live run lasts up to the full real-time window), stop the sync loop, pause audio.
function abortRun(){ if (ES){ ES.close(); ES=null; } stopSync(); const p=$('player'); if(p) p.pause(); LIVE_ON=false; SID=null; }

fetch('/api/cases').then(r=>r.json()).then(d=>{
  CASES = d.cases||[];
  ASR_OK = (d.asr === 'available');
  $('case').innerHTML = '<option value="">— custom text / audio upload —</option>' + CASES.map((c,i)=>
    `<option value="${i}">[${c.corpus}] ${c.title} (${c.label? 'harm':'benign'})</option>`).join('');
  $('opnote').innerHTML = `operating points: ${d.op.rule}; τ ${d.op.tau_low}/${d.op.tau_high}. `+
    `calib: ${d.calib}. ASR (wave-only): ${d.asr}.`;
});
$('case').onchange = () => {
  abortRun(); $('livebox').style.display='none';
  const c = CASES[$('case').value]; CUR = c || null;
  if (!c){ $('why').style.display='none'; return; }
  $('transcript').value = c.transcript; $('prosody').value = c.prosody;
  $('audio').value=''; AUDIO_B64=null; AUDIO_FILE=null; ASR_FILLED=false; $('asrbadge').style.display='none';
  $('player').style.display='none'; $('livewrap').style.display='none';
  $('why').style.display='block';
  $('why').innerHTML = `<b>${c.group}</b> · slice <code>${c.slice}</code> · canonical seed ${c.seed}<br>${c.why}`;
};
$('transcript').oninput = () => { ASR_FILLED=false; $('asrbadge').style.display='none'; if ($('case').value!==''){ $('case').value=''; CUR=null; $('why').style.display='none'; } };
$('audio').onchange = () => {
  const f = $('audio').files[0]; if(!f) return;
  abortRun();                       // stop any still-running stream before it clobbers the new input
  $('case').value=''; CUR=null; $('why').style.display='none';
  $('livebox').style.display='none';
  // Drop a previous file's auto-transcribed text so it does not linger as this file's input.
  if (ASR_FILLED){ $('transcript').value=''; ASR_FILLED=false; }
  $('asrbadge').style.display='none';
  AUDIO_FILE = f;
  if (PLAY_URL) URL.revokeObjectURL(PLAY_URL);
  PLAY_URL = URL.createObjectURL(f);
  const p = $('player'); p.src = PLAY_URL; p.style.display='block';
  // Live streaming needs auto-ASR (blank transcript) + faster-whisper installed.
  $('livewrap').style.display = ASR_OK ? 'flex' : 'none';
  const rd = new FileReader();
  rd.onload = () => { AUDIO_B64 = rd.result.split(',')[1]; AUDIO_NAME = f.name;
    $('opnote').innerHTML = `audio loaded: ${f.name} (${(f.size/1024).toFixed(0)} KB) — real prosody; auto-ASR if transcript blank`; };
  rd.readAsDataURL(f);
};
$('speed').oninput = () => $('spv').textContent = $('speed').value;

let L2INIT=false, RT_GPU=false;
function onL2Toggle(){
  const on = $('l2enable').checked;
  $('l2warn').style.display = (on && !RT_GPU) ? 'block' : 'none';
  if(!on){ $('l2load').disabled=true; $('l2score').disabled=true; $('l2rat').disabled=true; }
}
function pollStatus(){
  fetch('/api/status').then(r=>r.json()).then(d=>{
    const rt = d.runtime || {gpu:false, detail:'?'};
    RT_GPU = !!rt.gpu;
    $('runtime').textContent = (rt.gpu ? 'GPU — ' : 'CPU-only — ') + rt.detail;
    $('runtime').style.color = rt.gpu ? 'var(--benign)' : 'var(--esc)';
    const disabledBackend = d.l2.status === 'disabled';   // launched with --no-gate2
    if (!L2INIT){                    // default: L2 on when a GPU is present, off on CPU-only
      $('l2enable').checked = !!rt.gpu && !disabledBackend;
      $('l2enable').disabled = disabledBackend;
      L2INIT = true; onL2Toggle();
    }
    $('l2status').textContent = (disabledBackend ? 'disabled (--no-gate2)' :
      d.l2.status + (d.l2.load_s? ` (load ${d.l2.load_s}s)`:'') + (d.l2.device? ` [${d.l2.device}]`:''));
    $('l2model').textContent = d.l2.model || '–';
    const l2on = $('l2enable').checked && !disabledBackend;
    const ready = d.l2.status === 'ready';
    $('l2load').textContent = 'Load Gate-2 model' + (rt.gpu ? ' (GPU)' : ' (CPU)');
    $('l2load').disabled  = !l2on || ready || d.l2.status==='loading';
    $('l2score').disabled = !(l2on && ready && SID && DECIDED);
    $('l2rat').disabled   = !(l2on && ready && SID && DECIDED);
    if (d.l2.status==='unavailable') $('l2status').textContent='unavailable — '+d.l2.detail;
    sysUpdate(d.sys); gpuUpdate(d.gpu);
  }).catch(()=>{});
}
setInterval(pollStatus, 2000); pollStatus();

function sysUpdate(s){
  if(!s) return;
  // Normalize to a single 0–100 % scale (busy across all cores), not the per-core sum.
  const cpuN = s.cpu_pct==null ? null : Math.min(100, s.cpu_pct/s.ncpu);
  $('cpu').textContent = cpuN==null ? '–' : cpuN.toFixed(0)+' %';
  $('rss').textContent = s.rss_mb+' MB'; $('ncpu').textContent = s.ncpu;
  $('cpubar').style.width = Math.min(100,(cpuN||0))+'%';
  $('rssbar').style.width = Math.min(100,s.rss_mb/8192*100)+'%';
}
function gpuUpdate(g){
  if(!g){ $('gpu').textContent='no GPU'; $('gpubar').style.width='0%'; return; }
  $('gpu').textContent = g.util+' % · '+g.mem_mb+'/'+g.mem_total_mb+' MB'+(g.name? ' ('+g.name+')':'');
  $('gpubar').style.width = Math.min(100, g.util)+'%';
}

$('l2enable').onchange = onL2Toggle;
$('l2load').onclick = () => {
  $('l2load').disabled = true;                       // latch immediately; pollStatus takes over
  $('l2status').textContent = 'loading… (1.5B can take 30 s–minutes; watch the terminal)';
  fetch('/api/l2/load',{method:'POST',body:'{}'}).then(r=>r.json()).then(d=>{
    if (d && d.error) $('l2status').textContent = 'load error: ' + d.error;
  }).catch(e => { $('l2status').textContent = 'load request failed: ' + e; });
};

$('run').onclick = () => {
  const transcript = $('transcript').value.trim();
  if (!transcript && !AUDIO_B64) return alert('paste a transcript or upload an audio file');
  steps=[]; drawAll(); $('xai').textContent='…'; $('rat').textContent='–'; $('finalv').textContent='–';
  L2_DONE=false; DECIDED=false; $('asrbadge').style.display='none';
  $('prep').textContent=''; $('finalmap').textContent='';
  setLamp(null); $('l1lamp').textContent='running'; $('l1lamp').className='lamp on-esc pulse';
  $('l2lamp').textContent='idle'; $('l2lamp').className='lamp on-idle';
  // Live streaming: audio uploaded, transcript blank, ASR available, and the toggle on.
  const live = !!AUDIO_B64 && !transcript && ASR_OK && $('livemode') && $('livemode').checked;
  const body = { transcript, prosody:$('prosody').value, step_ms:+$('speed').value };
  if (AUDIO_B64){ body.audio_b64 = AUDIO_B64; body.audio_name = AUDIO_NAME; }
  // Playback starts on the 'start' event (after ASR loads), so the browser clock aligns with the
  // server's processing loop — otherwise the deadline marker drifts by the ASR load time.
  if (live){ body.stream = 'live'; stopSync(); FRONTIER=0; liveReset(); $('prep').textContent='loading streaming ASR…'; }
  else { $('livebox').style.display='none'; LIVE_ON=false; }
  if (ES){ ES.close(); ES=null; }   // close any previous run's stream before starting a new one
  fetch('/api/run',{method:'POST', body: JSON.stringify(body)}).then(r=>r.json()).then(d=>{
    if (d.error){ $('prep').textContent='error: '+d.error; $('l1lamp').textContent='error'; $('l1lamp').className='lamp on-harm'; stopSync(); return; }
    SID = d.sid;
    ES = new EventSource('/api/stream?sid='+SID);
    const myES = ES;
    ES.onmessage = m => { if (myES===ES) handle(JSON.parse(m.data), ES); };
  });
};

function liveReset(){
  $('livebox').style.display='block'; LIVE_ON=true;
  $('cl_play').textContent='0.0 s'; $('cl_proc').textContent='0.0 s'; $('cl_lag').textContent='–';
  $('rtf_asr').textContent='–'; $('rtf_miltl').textContent='–'; $('lagbar').style.width='0%';
  if (ASR_FILLED){ $('transcript').value=''; ASR_FILLED=false; }
}
function fmtT(s){ s=Math.max(0,Math.round(s||0)); const m=Math.floor(s/60), r=s%60; return m? (m+'m'+(r<10?'0':'')+r+'s') : (r+'s'); }
function startSync(){ stopSync(); const loop=()=>{ syncTick(); RAF=requestAnimationFrame(loop); }; RAF=requestAnimationFrame(loop); }
function stopSync(){ if (RAF){ cancelAnimationFrame(RAF); RAF=null; } }
function syncTick(){
  const p=$('player'); const play=p.currentTime||0;
  const dl = DEADLINE_S||0;
  // Once playback passes the decision deadline the verdict is already locked — clamp the
  // "processing vs playback" comparison to the deadline so the buffer reflects the decided window.
  const playC = dl? Math.min(play, dl) : play;
  $('cl_play').textContent = play.toFixed(1)+' s' + (dl && play>dl ? ' (past deadline)' : '');
  $('cl_proc').textContent = FRONTIER.toFixed(1)+' s';
  const lag = FRONTIER - playC;            // + = processing ahead of playback (buffered); − = falling behind
  $('cl_lag').textContent = (lag>=0? '+':'') + lag.toFixed(1)+' s '+(lag>=0?'(ahead)':'(behind)');
  const scale = Math.max(1, dl||DUR||1);
  $('lagbar').style.width = Math.min(100, Math.abs(lag)/scale*100)+'%';
  $('lagbar').style.background = lag>=0 ? 'var(--benign)' : 'var(--harm)';
  $('cl_lag').style.color = lag>=0 ? 'var(--benign)' : 'var(--harm)';
  // Verdict is already shown live once the anchor is met; if the device is behind, note we are settling.
  if (LIVE_ON && dl && play>=dl && FRONTIER < dl-0.05) $('cl_deadline').textContent='⏳ finalizing (device behind)…';
  // Stop the loop once processing has reached the deadline and playback is past it (or ended).
  const doneProc = FRONTIER >= (dl? dl-0.05 : DUR-0.05);
  if (!LIVE_ON && doneProc && (p.ended || (dl && play>=dl))) stopSync();
}

function handle(ev, es){
  sysUpdate(ev.sys);
  if (ev.type==='prep'){ $('prep').textContent = ev.msg; }
  else if (ev.type==='start'){
    nTotal = ev.n_nibbles;
    $('nnib').textContent = ev.n_nibbles+' × 8 s';
    $('anchor').textContent = ev.n_words+' words / anchor '+ev.anchor+(ev.anchor_ok?' ✓':' ✗');
    $('hasaudio').textContent = ev.prosody_real? ('REAL audio ('+ev.prosody+')') :
        (ev.has_audio? ('simulated ('+ev.prosody+')') : 'transcript-only (F-recentring)');
    if (ev.timings) $('t_feat').textContent = ev.timings.featurize_ms.toFixed(1)+' ms';
    if (ev.live){                              // live streaming: transcript builds up per chunk
      DUR = ev.duration_s||0; DEADLINE_S = ev.deadline_s||0;
      $('livebox').style.display='block'; LIVE_ON=true;
      $('asrdev').textContent = (ev.asr_device||'cpu').toUpperCase() + (ev.asr_note? ' ⚠':'');
      $('asrdev').title = ev.asr_note || '';
      if (ev.asr_note) $('prep').textContent = ev.asr_note;
      $('cl_deadline').textContent = fmtT(DEADLINE_S)+' (nibble '+ev.deadline_nibbles+')';
      if (ev.truncated){
        $('deadlinenote').style.display='block';
        $('deadlinenote').innerHTML = '⏱ Call '+fmtT(DUR)+' &gt; '+fmtT(DEADLINE_S)+' envelope — MiLTL '+
          'decides on the first '+fmtT(DEADLINE_S)+' (canonical); the rest is not scored.';
      } else { $('deadlinenote').style.display='none'; }
      $('transcript').value=''; ASR_FILLED=true; $('asrbadge').style.display='block';
      const p=$('player'); p.currentTime=0; p.play().catch(()=>{}); startSync();  // align clock to loop start
    } else {
      if (ev.transcript && !$('transcript').value.trim()){ $('transcript').value = ev.transcript; if (ev.asr) ASR_FILLED = true; }
      $('asrbadge').style.display = ev.asr ? 'block' : 'none';
    }
  } else if (ev.type==='step'){
    steps.push(ev); drawAll(); xmUpdate(ev);
    $('p1').textContent = ev.p1.toFixed(3); $('p1bar').style.width=(ev.p1*100)+'%';
    $('t_l1').textContent = ev.l1_ms.toFixed(1)+' ms'; setLamp(ev.band,true);
    if (ev.audio_time!=null){                  // live: advance the processing frontier + RTF + transcript
      FRONTIER = ev.audio_time;
      if (ev.transcript!=null) $('transcript').value = ev.transcript;
      $('anchor').textContent = (ev.transcript? ev.transcript.split(' ').filter(Boolean).length:0)+' words (streaming)';
      const asec = Math.max(1e-3, ev.audio_time);
      if (ev.asr_ms_tot!=null) $('rtf_asr').textContent = (ev.asr_ms_tot/1e3/asec).toFixed(2)+'× real-time';
      if (ev.miltl_ms_tot!=null) $('rtf_miltl').textContent = (ev.miltl_ms_tot/1e3/asec).toFixed(3)+'× real-time';
      if (ev.nibble_ms!=null){                 // per-nibble compute vs the 8 s budget (the edge signal)
        const budget = ev.budget_ms||8000, ok = !ev.over_budget;
        $('cl_budget').textContent = (ev.nibble_ms/1e3).toFixed(2)+' s / '+(budget/1e3).toFixed(0)+' s budget';
        $('cl_budget').style.color = ok ? 'var(--benign)' : 'var(--harm)';
        const behind = ev.behind_s||0;
        $('cl_keepup').textContent = ok
          ? (behind>0.3 ? 'catching up (backlog '+behind.toFixed(1)+' s)' : '✓ within budget')
          : '✗ over budget — falling behind ('+behind.toFixed(1)+' s)';
        $('cl_keepup').style.color = (ok && behind<=0.3) ? 'var(--benign)' : 'var(--harm)';
      }
      // Confirm the Gate-1 verdict live as soon as the observation anchor is met — no need to
      // wait for the loop to reach the 208 s deadline (a 3-min+ call decides mid-stream).
      if (ev.decidable){ liveVerdict(ev.band, ev.p1); if (ev.xai) $('xai').textContent = JSON.stringify(ev.xai,null,2); }
      else { $('l1lamp').textContent='observing ('+(ev.n_words||0)+'/'+(ev.anchor||80)+' words)'; $('l1lamp').className='lamp on-esc pulse'; }
    }
  } else if (ev.type==='final'){
    es.close();
    if (ev.live!=null || ev.asr===true) LIVE_ON=false;   // let the sync loop wind down after playback ends
    if (ev.band==='error' || ev.band==='undecidable'){
      LIVE_ON=false;
      $('l1lamp').textContent = ev.band; $('l1lamp').className='lamp '+(ev.band==='error'?'on-harm':'on-idle');
      $('xai').textContent = ev.note||''; setLamp(null); renderVerdicts(); return;
    }
    LIVE_ON=false;
    if (ev.asr_rtf!=null){
      $('rtf_asr').textContent = ev.asr_rtf.toFixed(2)+'× real-time'+
        (ev.compute_s!=null? ' (compute '+ev.compute_s+'s / '+ev.decided_s+'s window)':'');
      if (ev.miltl_rtf!=null) $('rtf_miltl').textContent = ev.miltl_rtf.toFixed(3)+'× real-time';
    }
    if (ev.decided_s!=null){                 // live: report the locked-at-deadline decision
      $('cl_deadline').textContent = '✓ decided at '+fmtT(ev.decided_s)+
        (ev.truncated? ' (of '+fmtT(ev.audio_s)+' call)' : '')+' — nibble '+ev.decided_nibbles;
    }
    const tot = steps.reduce((a,s)=>a+s.l1_ms,0);
    $('t_l1full').textContent = tot.toFixed(1)+' ms ('+steps.length+' steps)';
    $('p1').textContent=ev.p1.toFixed(3);
    $('xai').textContent = JSON.stringify(ev.xai,null,2);
    liveVerdict(ev.band, ev.p1, true);
    pollStatus(); renderVerdicts();
  }
}

// Show the confirmed Gate-1 verdict (Safe / Harm / Escalate→L2). Called live once the anchor is
// met, and again at loop end (final=true) as the settled result.
function liveVerdict(band, p1, final){
  DECIDED = true;                       // verdict is available → L2 buttons may be used now
  setLamp(band, band==='escalate' && !L2_DONE);
  $('l1lamp').textContent = final ? 'L1 decided' : 'L1 confirmed (live)';
  $('l1lamp').className = 'lamp on-benign';
  if (L2_DONE) return;                 // L2 already gave the final verdict — don't revert to escalate
  if (band==='escalate'){
    $('finalv').textContent='escalate'; $('finalmap').textContent='Gate-1 → needs L2';
    $('l2lamp').textContent='ESCALATED — score with L2'; $('l2lamp').className='lamp on-esc pulse';
  } else {
    $('finalv').textContent=p1.toFixed(3);
    $('finalmap').textContent = (band==='harm'?'HARM':'SAFE')+' — Gate-1 confident';
    $('l2lamp').textContent='not needed (confident)'; $('l2lamp').className='lamp on-idle';
  }
}

function setLamp(band, pulse){
  $('lampB').className='lamp'+(band==='benign'?' on-benign':'');
  $('lampE').className='lamp'+(band==='escalate'?' on-esc'+(pulse?' pulse':''):'');
  $('lampH').className='lamp'+(band==='harm'?' on-harm':'');
}

function renderVerdicts(){
  if (CUR){ renderCanonical(CUR); }
  else { renderLive(); }
}
function outcomeTag(o){
  const ok = (o==='TP'||o==='TN'); return `<span class="tag ${ok?'ok':'no'}">${o}${ok?' ✓':' ✗'}</span>`;
}
function renderCanonical(c){
  $('legtitle').textContent = 'Canonical benchmark verdicts (recorded)';
  const order = ['MiLTL-Cascade','hf-encoder','tree','cnn-bilstm','lexical-proxy','Bllossom-B3','Wave-Seq(audio-only)','MiLTL-Dual(naive fusion)'];
  const v = c.verdicts;
  let html = `<div class="small">Ground truth: <b>${c.label? 'HARM':'BENIGN'}</b> · slice <code>${c.slice}</code> · seed ${c.seed}</div>`;
  html += '<table class="tbl">';
  order.forEach(k=>{ if(!v[k]) return;
    const isM = k==='MiLTL-Cascade';
    html += `<tr class="${isM?'miltl-row':''}"><td>${isM?'<b>'+k+'</b>':k}</td>`+
            `<td style="text-align:right">${v[k].score.toFixed(3)}</td>`+
            `<td style="text-align:right">${outcomeTag(v[k].outcome)}</td></tr>`;
  });
  html += '</table>';
  $('legend2').innerHTML = html;
  $('legnote').textContent = 'From the 5-seed canonical run. TP/TN = correct, FP/FN = wrong. '+
    'The live trace above re-computes MiLTL with a demo-fit Calib, so its p1 illustrates the mechanism (may differ from the canonical p1).';
}
function renderLive(){
  $('legtitle').textContent = 'Detector verdicts (live, your input)';
  $('legend2').textContent = '…';
  fetch('/api/legacy',{method:'POST',body:JSON.stringify({sid:SID})}).then(r=>r.json()).then(d=>{
    if (d.error){ $('legend2').textContent=d.error; return; }
    const miltlBand = steps.length? steps[steps.length-1].band : '—';
    let html = `<table class="tbl"><tr class="miltl-row"><td><b>MiLTL (this demo)</b></td>`+
      `<td colspan=2 style="text-align:right"><span class="lamp on-${miltlBand==='harm'?'harm':(miltlBand==='benign'?'benign':'esc')}" style="padding:1px 8px">${miltlBand}</span></td></tr>`;
    d.rows.forEach(r=>{
      if (r.status){ html += `<tr><td>${r.name}</td><td colspan=2 style="text-align:right;color:var(--dim)">${r.status}</td></tr>`; return; }
      html += `<tr><td>${r.name}<br><span class="small">${r.detail}</span></td><td style="text-align:right">${r.score}</td>`+
        `<td style="text-align:right"><span class="lamp on-${r.verdict==='harm'?'harm':'benign'}" style="padding:1px 8px">${r.verdict}</span></td></tr>`;
    });
    html += '</table>';
    $('legend2').innerHTML = html;
    $('legnote').textContent = d.note;
  }).catch(()=>{ $('legend2').textContent='legacy panel unavailable'; });
}

$('l2score').onclick = () => {
  $('l2lamp').textContent='L2 scoring…'; $('l2lamp').className='lamp on-esc pulse';
  fetch('/api/l2/score',{method:'POST',body:JSON.stringify({sid:SID})}).then(r=>r.json()).then(d=>{
    if (d.error){ $('l2lamp').textContent=d.error; $('l2lamp').className='lamp on-harm'; return; }
    L2_DONE = true;                         // latch: streaming steps must not revert this to escalate
    const harm = d.p2 >= 0.5;
    setLamp(harm?'harm':'benign', false);   // final band reflects the L2 decision, not the escalate hold
    $('t_l2').textContent=d.latency_s+' s (1 forward)'; $('finalv').textContent=d.final.toFixed(3);
    $('finalmap').textContent=(harm?'HARM':'SAFE')+' · final '+d.final.toFixed(3)+' · P(harm)='+d.p2.toFixed(3);
    // Lamp shows the SAME final score as the center gauge (P(harm) is a different number — keep it out of the lamp).
    $('l2lamp').textContent='L2 done — '+(harm?'HARM':'SAFE')+' ('+d.final.toFixed(3)+')';
    $('l2lamp').className='lamp '+(harm?'on-harm':'on-benign'); sysUpdate(d.sys);
  });
};
$('l2rat').onclick = () => {
  $('rat').textContent='building rationale…';
  fetch('/api/l2/rationale',{method:'POST',body:JSON.stringify({sid:SID})}).then(r=>r.json()).then(d=>{
    if (d.error){ $('rat').textContent=d.error; return; }
    const sft = !!d.adapter;
    let txt = 'Reason (channel-grounded, always trusted): '+d.reason;
    if (d.slm) txt += '\n\nSLM note ('+d.model.split('/').pop()+(sft?' + SFT':'')+'): '+d.slm;
    else if (!sft) txt += '\n\n(SLM free-text omitted: it invented a scenario absent from the transcript (e.g. wiretapping) and was filtered — a zero-shot base-backbone limitation the SFT adapter fixes)';
    if (sft) txt += '\n\n✓ SFT adapter loaded — L2 is the released fine-tuned Gate-2.';
    else txt += '\n\n※ L2 verdict is a zero-shot BASE backbone (no SFT adapter): it can misjudge energetic-but-benign calls (e.g. teleshopping). The channel-grounded line above is the reliable part.';
    $('rat').textContent = txt;
    $('t_rat').textContent = d.latency_s+' s'; sysUpdate(d.sys);
  });
};

const COLORS={T:'#4cc9f0',I:'#f4a261',F:'#e76f51',E:'#e9c46a',X:'#c77dff'};
function drawAll(){ drawChannels(); drawRisk(); drawXM(); }
function drawXM(){
  const cv=$('xmchart'); if(!cv) return;
  const ctx=cv.getContext('2d'); cv.width=cv.clientWidth*2; cv.height=180; ctx.scale(2,2);
  const W=cv.clientWidth,H=90,PAD=26,PW=W-PAD,n=Math.max(nTotal,steps.length,1); ctx.clearRect(0,0,W,H);
  ctx.font='10px sans-serif';
  [0,0.5,1.0].forEach(y=>{ const yy=H*(1-y);
    ctx.strokeStyle='#2a3140';ctx.beginPath();ctx.moveTo(PAD,yy);ctx.lineTo(W,yy);ctx.stroke();
    ctx.fillStyle='#8b96a8';ctx.textAlign='right';ctx.fillText(y.toFixed(1),PAD-4,yy+3); });
  ctx.textAlign='left';
  const xw=i=>PAD+(i+0.5)/n*PW;
  // shade the gap where warm words exceed warm voice — that area IS the XM signal
  ctx.fillStyle='rgba(199,125,255,.20)';
  steps.forEach((s,i)=>{ const d=s.decomp; if(!d) return; if(d.warmth>d.V){
    const x=xw(i),w=Math.max(2,PW/n*0.8); ctx.fillRect(x-w/2,H*(1-d.warmth),w,H*(d.warmth-d.V)); }});
  // words-warmth line (green) and voice-warmth V line (blue)
  [['warmth','#2fbf71'],['V','#4cc9f0']].forEach(([key,col])=>{
    ctx.strokeStyle=col;ctx.lineWidth=1.8;ctx.beginPath();
    steps.forEach((s,i)=>{ if(!s.decomp)return; const x=xw(i),y=H*(1-(s.decomp[key]||0)); i?ctx.lineTo(x,y):ctx.moveTo(x,y);}); ctx.stroke(); });
  ctx.fillStyle='#8b96a8';ctx.fillText('🗣 words',PAD+4,10);ctx.fillText('🎙 voice',PAD+58,10);
}
function xmUpdate(ev){
  const d = ev.decomp; if(!d) return;
  const xm = (ev.nibble && ev.nibble.X!=null) ? ev.nibble.X : 0;
  $('xmw').style.width = Math.min(100, d.warmth*100)+'%';
  $('xmv').style.width = Math.min(100, d.V*100)+'%';
  $('xmwv').textContent = d.warmth.toFixed(2);
  $('xmvv').textContent = d.V.toFixed(2)+' (cold '+d.cold.toFixed(2)+')';
  $('xmval').textContent = xm.toFixed(3);
  $('xmval').style.color = xm>=0.2 ? 'var(--X)' : 'var(--dim)';
  const cues = ev.warm_cues || [];
  if (d.warmth>d.V && xm>=0.15){
    $('xmcue').style.display='block';
    $('xmcue').innerHTML = '⚠ warm words'+(cues.length? ' ('+cues.map(w=>'“'+w+'”').join(', ')+')':'')+
      ' spoken in a <b>cold voice</b> (V '+d.V.toFixed(2)+') → contradiction. Text-only would read this as friendly; audio-only hears only flatness — XM sees both.';
  } else { $('xmcue').style.display='none'; }
}
function drawChannels(){
  const cv=$('chart'),ctx=cv.getContext('2d'); cv.width=cv.clientWidth*2; cv.height=400; ctx.scale(2,2);
  const W=cv.clientWidth,H=200,PAD=26,PW=W-PAD,n=Math.max(nTotal,steps.length,1); ctx.clearRect(0,0,W,H);
  // Y axis: each channel (T/I/F/E/XM) is a normalized activation in 0–1; label the gridlines.
  ctx.font='10px sans-serif';
  [0,0.25,0.5,0.75,1.0].forEach(y=>{ const yy=H*(1-y);
    ctx.strokeStyle='#2a3140'; ctx.beginPath(); ctx.moveTo(PAD,yy); ctx.lineTo(W,yy); ctx.stroke();
    ctx.fillStyle='#8b96a8'; ctx.textAlign='right'; ctx.fillText(y.toFixed(2), PAD-4, yy+3); });
  ctx.textAlign='left';
  ['T','I','F','E','X'].forEach(k=>{ ctx.strokeStyle=COLORS[k];ctx.lineWidth=1.8;ctx.beginPath();
    steps.forEach((s,i)=>{const x=PAD+(i+0.5)/n*PW,y=H*(1-(s.nibble[k]||0)); i?ctx.lineTo(x,y):ctx.moveTo(x,y);}); ctx.stroke(); });
}
function drawRisk(){
  const cv=$('riskchart'),ctx=cv.getContext('2d'); cv.width=cv.clientWidth*2; cv.height=240; ctx.scale(2,2);
  const W=cv.clientWidth,H=120,PAD=26,PW=W-PAD,n=Math.max(nTotal,steps.length,1); ctx.clearRect(0,0,W,H);
  ctx.fillStyle='rgba(47,191,113,.10)';ctx.fillRect(PAD,H*0.60,PW,H*0.40);
  ctx.fillStyle='rgba(244,162,97,.08)';ctx.fillRect(PAD,H*0.10,PW,H*0.50);
  ctx.fillStyle='rgba(239,71,111,.10)';ctx.fillRect(PAD,0,PW,H*0.10);
  ctx.fillStyle='#8b96a8';ctx.font='10px sans-serif';ctx.textAlign='right';
  [['1.00',0],['0.90',0.10],['0.40',0.60],['0.00',1.0]].forEach(t=>ctx.fillText(t[0],PAD-4,H*t[1]+(t[1]===0?8:3)));
  ctx.textAlign='left';
  ctx.strokeStyle='#c77dff';ctx.lineWidth=2;ctx.beginPath();
  steps.forEach((s,i)=>{const x=PAD+(i+0.5)/n*PW,y=H*(1-s.p1); i?ctx.lineTo(x,y):ctx.moveTo(x,y);}); ctx.stroke();
  ctx.fillStyle='#8b96a8';
  ctx.fillText('harm ≥0.90',PAD+4,H*0.10-2); ctx.fillText('escalate',PAD+4,H*0.35); ctx.fillText('benign ≤0.40',PAD+4,H*0.60+10);
}
window.addEventListener('resize', drawAll);
</script>
</body></html>
"""

if __name__ == "__main__":
    main()
