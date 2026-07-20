"""MiLTL channel-bottleneck detector — inference adapter (see docs/ARCHITECTURE.md, docs/BENCHMARK.md) — production Gate-1.

Path: audio -> featurize_channels(prosody · text · speech-act · warmth) -> **trained extractors** -> [L,4] T/I/F/E ->
**dynamics synthesis** (front-loaded transient vs stationary; see docs/ARCHITECTURE.md) -> harm. In the head bake-off the
training-free dynamics synthesis (val AUROC 0.976) beat the trained head (0.96), so it is the default (≈0-param,
interpretable = edge identity). A trained head can be swapped in via --head.

Frozen: extractors and calibrator are fit on KorCCViD-train (freeze protocol). No tuning on KorMMP. Threshold = val calibration (intrinsic).
needs={"text"}: text always + prosody when audio is available (core). Below the anchor = undecidable -> 0.0 (safe).
"""
from __future__ import annotations

from pathlib import Path
from typing import FrozenSet

import numpy as np

from miltl.baseline.detector import BaselineDetector

_ANCHOR_WORDS = 210


def _agg3(tife, mask):
    idx = np.where(mask > 0.5)[0]
    if len(idx) == 0:
        z = np.zeros(4, np.float32); return z, z, z
    v = tife[idx]; k = max(1, len(idx) // 3)
    return v.mean(0), v[:k].mean(0), v[-k:].mean(0)


def _dynamics_score(tife, mask):
    """Dynamics synthesis (see docs/ARCHITECTURE.md): F+I−T + transient (early−late I·F). Fixed direction, training-free."""
    mean, early, late = _agg3(tife, mask)
    return float((mean[2] + mean[1] - mean[0]) + (early[1] - late[1]) + (early[2] - late[2]))


class ChannelBottleneckDetector(BaselineDetector):
    family = "multimodal"
    needs: FrozenSet[str] = frozenset({"text"})
    repro = "ok"
    exposes_channels = True

    def __init__(self, ckpt: str = "artifacts/models/channel_extractors.pt",
                 head: str = "", name: str = "MiLTL-Channel", device: str = "cpu",
                 op_thr: float = 0.0, score_scale: float = 0.5, channels: str = "calib",
                 xm_weight: float = 1.0, t_weight: float = 2.0, blend_analytic: float = 1.0,
                 anchor_words: int = 80, text_f_weight: float = 2.0, anchor_words_noaudio: int = 12,
                 codec_equalize: bool = False, text_f_baseline: float = 0.5):
        self.name = name
        self.device = device
        self._ckpt = ckpt
        self._head_ckpt = head
        self._channels = channels                          # calib = analytic channels (no text768 dependency) / extractor = trained extractors
        self._w_xm = xm_weight                             # XM (cross-modal = novelty) weight (see docs/BENCHMARK.md; default 1.0)
        self._w_t = t_weight                               # T (truth) weight — strongest channel in the 5-seed pool (dec 0.817) -> 2x (see docs/BENCHMARK.md)
        self._anchor = anchor_words                        # With audio: below anchor = insufficient (prosody) observation -> undecidable -> 0.0. 210 -> 80 (see docs/BENCHMARK.md)
        self._anchor_noaudio = anchor_words_noaudio        # Without audio: observation = transcript words (prosody irrelevant). Score short scams too -> lowered to 12 (see docs/BENCHMARK.md: anchor 80 gated 40% of harm = main culprit)
        self._w_f_text = text_f_weight                     # Audio-absence compensation (see docs/BENCHMARK.md): when E·XM (prosody) are disabled, reintroduce lexical coercion channel F (w=2)
        self._f0_text = text_f_baseline                    # F centering baseline (see docs/BENCHMARK.md): +w·(F−0.5) removes the constant offset -> normalizes banding
        self._codec_eq = codec_equalize                    # See docs/BENCHMARK.md: fair audio-channel equalization (telephone band + μ-law)
        self._blend = blend_analytic                       # Analytic-formula share (1.0 = pure analytic = canonical, 0 = pure head). Blended only when a head is loaded
        self._ext = None
        self._cal = None
        self._head = None
        self._enc = None
        self.intrinsic_op_thr = op_thr                     # Dynamics-synthesis threshold (val calibration). 0 = unset (AUROC-oriented)
        self._scale = score_scale
        self.intrinsic_src = f"동역학종합 임계={op_thr:.3f}(val)" if op_thr else "동역학종합(AUROC 위주, 임계 미설정)"
        self.notes = (f"채널병목({channels}→[L,4]→동역학종합). 앵커 {anchor_words}단어 미달=safe. "
                      f"head={'학습헤드' if head else '무학습 동역학종합'}.")

    def _load(self):
        if self._ext is not None:
            return
        import torch
        from miltl.native.channels import ChannelExtractors, ChannelConfig
        from miltl.native.channel_calib import Calib
        d = torch.load(self._ckpt, map_location=self.device, weights_only=False)
        cfg = ChannelConfig(**d["cfg"]) if isinstance(d.get("cfg"), dict) else ChannelConfig()
        m = ChannelExtractors(cfg).to(self.device).eval()
        m.load_state_dict(d["state"])
        self._ext = m
        self._cal = Calib.from_dict(d["calib"]) if "calib" in d else None
        if self._head_ckpt and Path(self._head_ckpt).is_file():
            hd = torch.load(self._head_ckpt, map_location=self.device, weights_only=False)
            from miltl.native.seq_head import TSMixerTiny, SeqHead
            self._head_in_ch = int(hd.get("in_ch", 4))
            self._head = (TSMixerTiny(in_ch=self._head_in_ch) if hd.get("kind") == "tsmixer"
                          else SeqHead(in_ch=self._head_in_ch)).to(self.device).eval()
            self._head.load_state_dict(hd["state"])
            if "calib" in hd:                              # Must match the calib used at head training time
                self._cal = Calib.from_dict(hd["calib"])
        if self._channels == "extractor":                  # Analytic channels (calib) have no text768 dependency -> klue-roberta not needed
            from miltl.native.features import KlueRobertaEncoder
            self._enc = KlueRobertaEncoder(device=self.device)
        # Self-verification banner — instantly identifies a stale checkout. If this line is missing from logs or the coefficients differ, it's old code.
        import sys
        mode = f"blend={self._blend:.2f}·head" if (self._head is not None and self._blend < 1.0) else "순해석식"
        print(f"[MiLTL] canonical scorer: risk = E - {self._w_t:.1f}*T + I + {self._w_xm:.1f}*XM  "
              f"({mode}, anchor={self._anchor})", file=sys.stderr, flush=True)

    def fit(self, train_calls) -> None:
        return None

    def _cross_modal(self, nci):
        """Cross-modal diagnostics: valid means of (XM, V_prosody, warmth) — per-call decomposition of the novelty signal."""
        if self._cal is None:
            return 0.0, 0.5, 0.0
        from miltl.native.channel_calib import avd_from_z, evidence, _IX
        z = self._cal.zfeat(nci.prosody)
        avd = avd_from_z(z)
        w = nci.warmth if nci.warmth is not None else np.zeros(len(nci.mask), np.float32)
        ev = evidence(avd, nci.speech_act, w, z[:, _IX["pause_ratio"]])
        m = nci.mask > 0.5
        if not m.any():
            return 0.0, 0.5, 0.0
        return float(ev["XM"][m].mean()), float(avd[m, 1].mean()), float(w[m].mean())

    def score(self, call) -> float:
        tr = getattr(call, "transcript", "") or ""
        # Anchor = sufficient-observation gate. With audio = prosody observation (80 words); without = transcript observation (12 words, scores short scams).
        # See docs/BENCHMARK.md: on audio-free KorCCViD the 80-word anchor dumped 40% of harm (short scams) to 0.0 = main cause of AUROC 0.43.
        anchor = self._anchor if getattr(call, "audio_uri", None) else self._anchor_noaudio
        if len(tr.split()) < anchor:
            # Undecidable (insufficient observation) = 0.0. Refresh last_diag with a gated marker (prevents channel contamination from the previous call).
            self.last_diag = {"T": 0.0, "I": 0.0, "F": 0.0, "E": 0.0, "XM": 0.0,
                              "V": 0.5, "cold": 0.5, "warmth": 0.0, "audio": 0, "gated": 1}
            return 0.0
        self._load()
        import torch
        from miltl.native.nibble_features import featurize_channels
        pcm = None
        uri = getattr(call, "audio_uri", None)
        if uri:
            try:
                from miltl.nibble.audio_decode import decode_to_pcm
                pcm, _ = decode_to_pcm(uri, sr=16000)
            except Exception:                              # noqa: BLE001
                pcm = None
        nci = featurize_channels(pcm, tr.split(), text_enc=self._enc,   # enc=None (calib) -> Mock (text unused)
                                 codec_equalize=self._codec_eq)
        if self._channels == "calib":
            from miltl.native.channel_calib import channels as _analytic
            tife = _analytic(nci, self._cal)               # Analytic membership functions (prosody+lexical+XM, no text768 dependency)
        else:
            sa5 = np.concatenate([nci.speech_act, (nci.warmth if nci.warmth is not None
                                                   else np.zeros(len(nci.mask), np.float32))[:, None]], -1)
            with torch.no_grad():
                tife, _ = self._ext(torch.tensor(nci.prosody[None], device=self.device),
                                    torch.tensor(nci.text[None], device=self.device),
                                    torch.tensor(sa5[None].astype(np.float32), device=self.device))
                tife = tife[0].cpu().numpy()
        mean, _, _ = _agg3(tife, nci.mask)
        xm, v_pros, warmth = self._cross_modal(nci)        # Cross-modal contradiction (main novelty signal) + V·warmth decomposition
        self.last_diag = {"T": float(mean[0]), "I": float(mean[1]), "F": float(mean[2]), "E": float(mean[3]),
                          "XM": xm, "V": v_pros, "cold": 1.0 - v_pros, "warmth": warmth,
                          "audio": 1 if pcm is not None else 0}
        # Neutrosophic-affective risk (see docs/BENCHMARK.md): risk = E − w_t·T + I + w_xm·XM.
        # PAD threat signals: E (arousal) up − T (truth) down + I (latent indeterminacy) up + XM (cross-modal = novelty legacy cannot do).
        # F (overt coercion) carries no signal on this distribution (AUROC 0.506), so it is excluded — it was the main cause of FPs on calm real call-center audio.
        # 5-seed pool (500 calls): T is the strongest channel (dec 0.817) -> w_t=2. decorrelated 0.777 -> 0.798, F1 0.759 -> 0.768 (±0.032).
        risk = float(mean[3]) - self._w_t * float(mean[0]) + float(mean[1]) + self._w_xm * xm
        # Audio-absence compensation (see docs/BENCHMARK.md): when prosody-dependent E (arousal) and XM (cross-modal) are disabled
        # (prosody=0), the risk formula collapses to −2T+I -> L1 banding unreliable on transcript-only corpora (KorCCViD). In that case
        # the lexical coercion channel F (threat/command/subversion vocabulary) is reintroduced to compensate — the reason F was dropped
        # from the canonical formula (prosody FPs on calm real call-center audio) cannot occur when there is no audio.
        if pcm is None:
            # F centering (see docs/BENCHMARK.md): calm call-center benign also has high F (~0.5) -> uncentered +w·F is a constant
            # offset (~1.0) that saturates both benign and harm (sigmoid≈0.85), collapsing the bands (everything harm) and preventing
            # L2 intervention. Centering at F0=0.5 keeps only the discriminative slope: AUROC preserved (0.870) + banding normalized
            # (benign/escalate/harm distribution recovered).
            risk += self._w_f_text * (float(mean[2]) - self._f0_text)
        p_analytic = float(1.0 / (1.0 + np.exp(-(risk - self.intrinsic_op_thr) / self._scale)))  # Monotone mapping
        # Canonical = pure analytic formula (blend=1.0): E−T+I+XM beat the trained head (F1 0.648) with F1 0.725 and decorr
        # 0.773 on smoke_42. The head is only mixed in as an option (blend<1) — no over-suppression of XM on high-T natural phishing.
        if self._head is not None and self._blend < 1.0:
            if getattr(self, "_head_in_ch", 4) == 5:       # Trained head takes [L,5]=(T,I,F,E,XM)
                from miltl.native.channel_calib import channels5
                feat = channels5(nci, self._cal)
            else:
                feat = tife
            with torch.no_grad():
                p = float(torch.sigmoid(self._head(torch.tensor(feat[None], device=self.device),
                                                   torch.tensor(nci.mask[None], device=self.device)))[0])
            return (1.0 - self._blend) * p + self._blend * p_analytic
        return p_analytic


class MiLTLCascadeDetector(BaselineDetector):
    """The complete MiLTL product = L1 (channel Gate-1) -> band -> L2 (Gate-2 SLM, escalate only) seamless cascade, as a single detector.

    score(call) = final cascade harm score (continuous): confident bands keep the Gate-1 p1; escalate uses the Gate-2 SLM's
    **continuous P(harm)** (score_channels logits, replacing the earlier binary 0.9/0.1 -> restores in-band ordering = AUROC).
    Even without audio (transcript-only KorCCViD) L1/L2 run seamlessly as usual, with L1 filling the gap left by disabled
    prosody (E·XM) via **reintroduction of the lexical coercion channel F** (ChannelBottleneckDetector.text_f_weight,
    see docs/BENCHMARK.md) so banding is preserved.
    MiLTL is computed **exactly once** here -> enters the same bench pass as the legacy systems.
    last_diag = channels (T/I/F/E/XM/cold/warmth) + p1·band·decision (post-hoc analysis, XAI evidence).
    """

    family = "multimodal"
    needs: FrozenSet[str] = frozenset({"text"})
    repro = "ok"
    exposes_channels = True

    def __init__(self, ckpt: str = "artifacts/models/channel_extractors.pt", head: str = "",
                 name: str = "MiLTL-Cascade", device: str = "cpu",
                 gate2_model: str = "Qwen/Qwen2.5-1.5B-Instruct", gate2_adapter: str = "",
                 tau_low: float = 0.40, tau_high: float = 0.90, anchor_words: int = 80,
                 codec_equalize: bool = False):
        self.name = name
        self.device = device
        self._tau_low, self._tau_high = tau_low, tau_high
        self._g2_model, self._g2_adapter = gate2_model, gate2_adapter
        self._l1 = ChannelBottleneckDetector(ckpt=ckpt, head=head, device=device, channels="calib",
                                             blend_analytic=1.0, anchor_words=anchor_words,
                                             codec_equalize=codec_equalize)
        self._g2 = None
        self.notes = f"L1(E−2T+I+XM) → escalate(τ {tau_low}~{tau_high}) → L2 SLM({gate2_model})"

    def fit(self, train_calls) -> None:
        return None

    def _load_g2(self):
        if self._g2 is not None:
            return
        from adapters.baselines.gate2_slm import Gate2SLM
        g2 = Gate2SLM(model_name=self._g2_model, finetune=False)
        if self._g2_adapter:
            try:
                g2.load_adapter(self._g2_adapter)
            except ImportError:
                import sys
                print("[MiLTL-Cascade] [WARN] peft 미설치 → Gate-2 zero-shot 폴백 (`pip install peft`)",
                      file=sys.stderr, flush=True)
                g2.fit([])
        else:
            g2.fit([])
        self._g2 = g2

    def score(self, call) -> float:
        tr = getattr(call, "transcript", "") or ""
        p1 = float(self._l1.score(call))                    # L1 (Gate-1, lexical F compensation when audio absent) — computed only here
        d = dict(getattr(self._l1, "last_diag", {}))
        l2 = None
        if p1 <= self._tau_low:                             # Seamless banding (same path with or without audio)
            band, decision, final = "benign", "benign", p1
        elif p1 >= self._tau_high:
            band, decision, final = "harm", "harm", p1
        else:                                               # escalate -> L2 (Gate-2 SLM), continuous P(harm)
            self._load_g2()
            from adapters.baselines.gate2_slm import summarize_channels
            summary = summarize_channels(d)                 # Channel summary MiLTL passed to the LMM (report sheet, reproducibility)
            prompt = self._g2.prompt_channels(tr, d)        # Exact prompt (same input = same decision = reproducible)
            p2 = float(self._g2.score_channels(tr, d))      # L2 final P(harm) — logit-based, continuous
            decision = "harm" if p2 >= 0.5 else "benign"
            # Map the L2 P(harm) into the escalate band's numeric range [τ_low, τ_high] -> globally monotone with the confident bands (p1).
            # Mixed scales (confident=p1, escalate=P(harm)) scrambled ranking at band boundaries -> fixed the Cascade<L1 issue (see docs/BENCHMARK.md).
            final = self._tau_low + p2 * (self._tau_high - self._tau_low)
            band = "escalate"
            l2 = {"p2": round(p2, 4), "summary": summary, "prompt": prompt}
        # XAI deterministic reason and action (user-facing + paper post-hoc analysis). _underscore keys = report-sheet only (not written to CSV).
        from miltl.native.explain import explain_decision
        xai = explain_decision(d, decision, tr, p1)
        self.last_diag = {**d, "p1": round(p1, 4), "band": band, "decision": decision,
                          "_l2": l2, "_xai": xai}
        return float(final)
