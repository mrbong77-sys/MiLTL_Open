"""Gate-2 tiny-SLM final harm arbiter (see docs/ARCHITECTURE.md) — PoC backbone Qwen2.5-0.5B-Instruct.

Final arbiter of the MiLTL cascade: it receives only the **borderline windows** where Gate-1 (lightweight,
always-on) crossed τ_alert, and confirms final harm from the transcript window + a **nibble/prosody fusion
summary** (T/I/F/E occupancy, F persistence, F∧E co-occurrence, harm ramp, availability state, Gate-1 p1).
Multimodality is already fused into the nibble byte -> the fusion context is **injected as text** into a
text LM (no audio-native model needed).

Scoring: **sum of token log-probs** over the yes/no families (robust to byte-BPE; same technique as
bllossom_llm). With fit(finetune=True), LoRA SFT on borderline windows + hard negatives. Heavy deps
(torch/transformers/peft) are lazily imported -> importing this module is stdlib-safe.
Swapping backbones is just a model_name swap (reused for the LFM2.5-230M and Gemma-3-270M ablations).

  # PoC: rather than using it directly via run_baselines, integrate as the MiLTL cascade Gate-2. Standalone score:
  Gate2SLM().score_window(transcript, text_nibbles, wave_nibbles, p1=0.62)
"""
from __future__ import annotations

from typing import List, Optional, Sequence

_INSTALL_HINT = ("torch/transformers(+peft for LoRA) 필요 — pip install -r adapters/baselines/requirements.txt (DGX)")

_SYS = ("당신은 보이스피싱 최종 판정관입니다. 아래 통화 윈도우(전사)와 신호 요약(위협 T·불확정 I·유해 F·"
        "위협에너지 E 의 2분 누적 패턴)을 근거로, 이 통화가 **실제 위해(harm: 자금 이체·개인정보 탈취·협박)**로 "
        "이어지는 보이스피싱인지 판정하세요. 정상 금융/업무 상담(계좌·대출 어휘가 있어도 협박·이체유도 없음)은 "
        "'아니오'. 반드시 '예' 또는 '아니오'로만 답하세요.")


def _bits(n: int):
    return ((n >> 3) & 1, (n >> 2) & 1, (n >> 1) & 1, n & 1)   # Layout T I F E (T=MSB)


def _chan_features(nibs):
    vals = [n for n in nibs if n is not None]
    if not vals:
        return None
    B = [_bits(n) for n in vals]
    occ = [sum(b[i] for b in B) / len(B) for i in range(4)]       # T I F E occupancy
    fe = sum(1 for b in B if b[2] and b[3]) / len(B)              # F∧E co-occurrence
    run = mx = 0                                                  # Max F persistence (run-length)
    for b in B:
        run = run + 1 if b[2] else 0; mx = max(mx, run)
    half = max(1, len(B) // 2)                                    # harm ramp: late-half F − early-half F
    ramp = (sum(b[2] for b in B[half:]) / max(1, len(B) - half)
            - sum(b[2] for b in B[:half]) / half)
    return {"T": occ[0], "I": occ[1], "F": occ[2], "E": occ[3],
            "fe": fe, "f_run": mx, "ramp": ramp, "n": len(B)}


def nibble_features(text_nibbles: Sequence, wave_nibbles: Sequence,
                    p1: Optional[float] = None, state: Optional[str] = None) -> dict:
    """Nibble streams -> feature dict (shared by TQA situation selection and the summary; pure, DGX-free)."""
    return {"text": _chan_features(text_nibbles), "wave": _chan_features(wave_nibbles),
            "p1": p1, "state": state}


def summarize_nibbles(text_nibbles: Sequence, wave_nibbles: Sequence,
                      p1: Optional[float] = None, state: Optional[str] = None) -> str:
    """Nibble streams -> Korean fusion summary for the Gate-2 prompt (pure, DGX-free)."""
    feat = nibble_features(text_nibbles, wave_nibbles, p1, state)
    parts = []
    for nm, key in (("텍스트", "text"), ("파형", "wave")):
        c = feat[key]
        if c is None:
            parts.append(f"{nm}: (결측)"); continue
        parts.append(f"{nm}: T{c['T']:.2f} I{c['I']:.2f} F{c['F']:.2f} E{c['E']:.2f} · "
                     f"F지속max {c['f_run']}/{c['n']} · F∧E공기 {c['fe']:.2f} · harm기울기 {c['ramp']:+.2f}")
    if p1 is not None:
        parts.append(f"Gate-1 p1 {p1:.2f}")
    if state:
        parts.append(f"가용성 {state}")
    return " · ".join(parts)


def summarize_channels(diag: dict) -> str:
    """Channel-bottleneck diagnostics (T/I/F/E/XM) -> **raw signals** for the Gate-2 prompt (no verdict included).

    XM is stated explicitly as an auxiliary signal that transcript-only LLMs like Bllossom cannot see, but
    **no conclusion (benign/phishing) is included** — fixes the issue where the small SLM parroted the
    summary's conclusion and missed low-XM phishing (see docs/BENCHMARK.md).
    """
    xm = float(diag.get("XM", 0.0)); cold = float(diag.get("cold", 0.0)); warmth = float(diag.get("warmth", 0.0))
    T = float(diag.get("T", 0.0)); I = float(diag.get("I", 0.0))
    F = float(diag.get("F", 0.0)); E = float(diag.get("E", 0.0))
    return (f"교차모달 모순 XM={xm:.2f} (표현 따뜻함 {warmth:.2f} vs 음성 냉정도 {cold:.2f}; "
            f"XM 클수록 '따뜻한 말·냉정한 음성' 불일치=기만 가능성↑, 낮아도 전사에 사칭·이체유도 있으면 피싱). "
            f"뉴트로소픽 참T={T:.2f}·잠복위협I={I:.2f}·표출강압F={F:.2f}·각성E={E:.2f}. "
            f"※ 이는 보조 지표이며 최종 판정은 아래 전사 내용을 우선 근거로 하세요.")


_SYS_CH = ("당신은 보이스피싱 최종 판정관입니다. **통화 전사 내용을 최우선 근거**로 판정하세요: "
           "기관(검찰·경찰·금융감독원·은행)·가족 사칭, 계좌이체·안전계좌·본인인증 유도, 개인정보/OTP 요구, "
           "협박·압박·환급/대출 미끼가 있으면 피싱('예')입니다. 보조로 음성-언어 교차모달 신호(XM)를 참고하되, "
           "**XM 이 낮아도 전사에 위 사칭·이체유도가 있으면 반드시 '예'**. 협박·이체유도 없는 정상 상담만 '아니오'. "
           "요약을 복창하지 말고 전사를 읽고 판단하세요. 반드시 '예'(피싱) 또는 '아니오'(정상)로 답하세요.")


class Gate2SLM:
    """Gate-2 final-arbiter SLM. PoC = Qwen2.5-0.5B-Instruct; swap model_name for ablations."""

    name = "Gate2-SLM(Qwen2.5-0.5B)"

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
                 finetune: bool = False, load_8bit: bool = False, max_len: int = 1024,
                 epochs: int = 2, lr: float = 2e-4, seed: int = 20260706):
        self.model_name = model_name
        self.finetune, self.load_8bit = finetune, load_8bit
        self.max_len, self.epochs, self.lr, self.seed = max_len, epochs, lr, seed
        self._torch = self._tok = self._model = None
        self._pos_ids = self._neg_ids = None

    # ---- Lazy backend ----
    def _ensure(self):
        if self._torch is not None:
            return
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError as e:
            raise RuntimeError(_INSTALL_HINT) from e
        self._torch, self._AutoTok, self._AutoLM = torch, AutoTokenizer, AutoModelForCausalLM

    def _first_ids(self, surfaces):
        ids = set()
        for s in surfaces:
            enc = self._tok.encode(s, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        return sorted(ids)

    def _build(self):
        torch = self._torch
        self._tok = self._AutoTok.from_pretrained(self.model_name)
        kw = {"device_map": "auto"}
        if self.load_8bit:
            try:
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except ImportError:
                pass
        elif self.finetune:
            # fp16 backward is unstable for training (NaN) -> prefer bf16 (wider range), fp32 when unsupported.
            bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
            kw["torch_dtype"] = torch.bfloat16 if bf16 else torch.float32
        else:
            kw["torch_dtype"] = torch.float16                # Inference in fp16 (memory, speed)
        self._model = self._AutoLM.from_pretrained(self.model_name, **kw)
        self._pos_ids = self._first_ids(["예", " 예", "네", " 네", "위험", " 위험", "유해", "Yes", " Yes"])
        self._neg_ids = self._first_ids(["아니", " 아니", "아니오", " 아니오", "정상", " 정상", "안전", "No", " No"])
        ov = set(self._pos_ids) & set(self._neg_ids)
        self._pos_ids = [i for i in self._pos_ids if i not in ov]
        self._neg_ids = [i for i in self._neg_ids if i not in ov]

    def prompt(self, transcript: str, summary: str, tqa: Optional[Sequence[str]] = None) -> str:
        tqa_block = ("\n[점검 질문(TQA)]\n" + "\n".join(f"- {q}" for q in tqa)) if tqa else ""
        msgs = [{"role": "system", "content": _SYS},
                {"role": "user", "content": f"[신호 요약]\n{summary}{tqa_block}\n\n[통화 전사]\n"
                                            f"{(transcript or '')[:self.max_len]}\n\n판정(예/아니오):"}]
        try:
            return self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:  # noqa: BLE001
            return f"{_SYS}\n[신호 요약]\n{summary}\n[통화 전사]\n{(transcript or '')[:self.max_len]}\n판정(예/아니오):"

    # ---- Training (borderline windows + hard negatives) ----
    def fit(self, examples: Sequence[dict]) -> None:
        """examples: [{transcript, text_nibbles, wave_nibbles, p1?, state?, label(0/1)}...]. LoRA SFT when finetune=True."""
        self._ensure(); self._build()
        if not self.finetune:
            return
        labels = [int(e["label"]) for e in examples]
        if not examples or sum(labels) in (0, len(labels)):
            return
        torch = self._torch
        from peft import LoraConfig, get_peft_model
        self._model = get_peft_model(self._model, LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"]))
        opt = torch.optim.AdamW([p for p in self._model.parameters() if p.requires_grad], lr=self.lr)
        self._model.train()
        rng = __import__("random").Random(self.seed)
        idx = list(range(len(examples)))
        for _ in range(self.epochs):
            rng.shuffle(idx)
            for i in idx:
                e = examples[i]
                pr = self.prompt(e["transcript"], summarize_nibbles(
                    e.get("text_nibbles", []), e.get("wave_nibbles", []), e.get("p1"), e.get("state")))
                ans = " 예" if e["label"] else " 아니오"
                enc = self._tok(pr + ans, return_tensors="pt", truncation=True,
                                max_length=self.max_len).to(self._model.device)
                plen = len(self._tok(pr, truncation=True, max_length=self.max_len)["input_ids"])
                lab = enc["input_ids"].clone(); lab[0, :plen] = -100      # Supervise the answer only
                out = self._model(**enc, labels=lab)
                out.loss.backward(); opt.step(); opt.zero_grad()

    def fit_channels(self, examples: Sequence[dict]) -> None:
        """LoRA SFT on channel-bottleneck escalate examples (see docs/BENCHMARK.md). examples: [{transcript, diag, label}].

        Supervises '예/아니오' with the **same prompt as score_channels (prompt_channels)** -> train/inference prompt
        consistency (previously trained with _judge_prompt (CoT) but inferred with prompt_channels = mismatch ->
        wasted adapter; fixed, see docs/BENCHMARK.md).
        Learns to discriminate low-XM phishing vs calm benign in the escalate band (fixes the zero-shot 0.5B
        parroting problem). Frozen: KorCCViD + synthetic.
        """
        self._ensure(); self._build()
        labels = [int(e["label"]) for e in examples]
        if not examples or sum(labels) in (0, len(labels)):
            print("[Gate2SFT] cannot train (single class / empty set)"); return
        torch = self._torch
        from peft import LoraConfig, get_peft_model
        self._model = get_peft_model(self._model, LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"]))
        opt = torch.optim.AdamW([p for p in self._model.parameters() if p.requires_grad], lr=self.lr)
        self._model.train()
        rng = __import__("random").Random(self.seed)
        idx = list(range(len(examples)))
        for ep in range(self.epochs):
            rng.shuffle(idx); tot = 0.0; nb = 0; skipped = 0
            for i in idx:
                e = examples[i]
                pr = self.prompt_channels(e["transcript"], e["diag"])   # Same prompt as score_channels
                ans = " 예" if e["label"] else " 아니오"
                # Concatenate explicitly so the answer tokens are never truncated away, which would make all labels -100 (-> NaN).
                ans_ids = self._tok(ans, add_special_tokens=False)["input_ids"]
                pr_ids = self._tok(pr, truncation=True,
                                   max_length=self.max_len - len(ans_ids) - 1)["input_ids"]
                ids = pr_ids + ans_ids
                input_ids = torch.tensor([ids], device=self._model.device)
                lab = torch.tensor([[-100] * len(pr_ids) + ans_ids], device=self._model.device)
                out = self._model(input_ids=input_ids,
                                  attention_mask=torch.ones_like(input_ids), labels=lab)
                loss = out.loss
                if not torch.isfinite(loss):                # NaN/Inf guard — skip this batch
                    opt.zero_grad(); skipped += 1; continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(              # Gradient clipping (prevents blow-up)
                    [p for p in self._model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad(); tot += float(loss.detach()); nb += 1
            print(f"[Gate2SFT] epoch {ep+1}/{self.epochs} loss={tot/max(nb,1):.3f} "
                  f"(batches {nb}, NaN-skipped {skipped})", flush=True)
        self._model.eval()

    def load_adapter(self, path: str) -> None:
        """Load a saved LoRA adapter (for inference). Frozen weights + trained adapter."""
        self._ensure(); self._build()
        from peft import PeftModel
        self._model = PeftModel.from_pretrained(self._model, path)
        self._model.eval()

    # ---- Inference ----
    def score(self, transcript: str, summary: str, tqa: Optional[Sequence[str]] = None) -> float:
        if self._model is None:
            return 0.0
        torch = self._torch
        self._model.eval()
        with torch.no_grad():
            enc = self._tok(self.prompt(transcript, summary, tqa), return_tensors="pt",
                            truncation=True, max_length=self.max_len).to(self._model.device)
            probs = torch.softmax(self._model(**enc).logits[0, -1].float(), dim=-1)
            pos, neg = probs[self._pos_ids].sum(), probs[self._neg_ids].sum()
            denom = (pos + neg).item()
        return (pos.item() / denom) if denom > 1e-9 else 0.5           # P(harm)

    def score_window(self, transcript, text_nibbles, wave_nibbles, p1=None, state=None,
                     kw_present=None, use_tqa: bool = True) -> float:
        """Sensor nibbles -> fusion summary + situation-specific TQA -> Gate-2 harm score (see docs/ARCHITECTURE.md)."""
        from .tqa import select_tqa
        feat = nibble_features(text_nibbles, wave_nibbles, p1, state)
        summary = summarize_nibbles(text_nibbles, wave_nibbles, p1, state)
        tqa = select_tqa(feat, kw_present=kw_present) if use_tqa else None
        return self.score(transcript, summary, tqa)

    def prompt_channels(self, transcript: str, diag: dict) -> str:
        """Channel-bottleneck diagnostics (XM up front) + transcript -> Gate-2 prompt (cross-modal arbiter system message)."""
        summary = summarize_channels(diag)
        msgs = [{"role": "system", "content": _SYS_CH},
                {"role": "user", "content": f"[통화 전사]\n{(transcript or '')[:self.max_len]}\n\n"
                                            f"[보조 신호(참고용)]\n{summary}\n\n판정(예/아니오):"}]
        try:
            return self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:  # noqa: BLE001
            return f"{_SYS_CH}\n[교차모달 신호 요약]\n{summary}\n[통화 전사]\n{(transcript or '')[:self.max_len]}\n판정(예/아니오):"

    def _judge_prompt(self, transcript: str, diag: dict):
        """Build the judgment prompt (transcript first + raw channel signals). Shared by judge_channels and fit_channels (no generation)."""
        summary = summarize_channels(diag)
        sys2 = _SYS_CH + " 먼저 전사에서 사칭·이체유도·협박 여부를 1~2문장으로 짚고, 마지막 줄에 '판정: 예' 또는 '판정: 아니오'로 끝내세요."
        msgs = [{"role": "system", "content": sys2},
                {"role": "user", "content": f"[통화 전사]\n{(transcript or '')[:self.max_len]}\n\n"
                                            f"[보조 신호(참고용)]\n{summary}"}]
        try:
            prompt = self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:  # noqa: BLE001
            prompt = f"{sys2}\n[전사]\n{(transcript or '')[:self.max_len]}\n[보조]\n{summary}\n"
        return prompt, summary

    def judge_channels(self, transcript: str, diag: dict, max_new_tokens: int = 160) -> dict:
        """Generation-based final judgment (with rationale CoT) — **for the report sheet**. Returns the prompt, rationale, and decision for post-hoc analysis and reproducibility.

        Returns dict: {decision('harm'|'benign'), rationale(generated reasoning), raw(full generation), summary(injected summary),
        prompt(full prompt)}. Called only for the escalate band. Decision parses '판정: 예/아니오' (falls back to the more frequent of 예/아니오).
        """
        prompt, summary = self._judge_prompt(transcript, diag)
        if self._model is None:                              # Not loaded (test/dry run) — record only summary and prompt
            return {"decision": None, "rationale": "", "raw": "", "summary": summary, "prompt": prompt}
        torch = self._torch
        self._model.eval()
        with torch.no_grad():
            enc = self._tok(prompt, return_tensors="pt", truncation=True,
                            max_length=self.max_len).to(self._model.device)
            out = self._model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                       repetition_penalty=1.3, no_repeat_ngram_size=3,
                                       pad_token_id=self._tok.eos_token_id)
            raw = self._tok.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        import re
        m = re.search(r"판정\s*[:：]?\s*(예|아니오|아니요)", raw)
        if m:
            decision = "harm" if m.group(1) == "예" else "benign"
        else:                                                # Fallback: more frequent of 예/아니오
            decision = "harm" if raw.count("예") > raw.count("아니") else "benign"
        rationale = raw.split("판정")[0].strip()[:400]
        return {"decision": decision, "rationale": rationale, "raw": raw, "summary": summary, "prompt": prompt}

    def score_channels(self, transcript: str, diag: dict) -> float:
        """Gate-2 final harm decision based on channel-bottleneck Gate-1 diagnostics (XM injected up front). Called only for the escalate band."""
        if self._model is None:
            return 0.5
        torch = self._torch
        self._model.eval()
        with torch.no_grad():
            enc = self._tok(self.prompt_channels(transcript, diag), return_tensors="pt",
                            truncation=True, max_length=self.max_len).to(self._model.device)
            probs = torch.softmax(self._model(**enc).logits[0, -1].float(), dim=-1)
            pos, neg = probs[self._pos_ids].sum(), probs[self._neg_ids].sum()
            denom = (pos + neg).item()
        return (pos.item() / denom) if denom > 1e-9 else 0.5

    def save(self, path: str) -> None:
        self._model.save_pretrained(path); self._tok.save_pretrained(path)  # GGUF export is separate (llama.cpp)
