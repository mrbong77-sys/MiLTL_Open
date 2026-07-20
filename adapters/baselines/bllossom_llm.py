"""B3 — on-device small-LLM baseline (docs/BASELINES.md; Sim & Kim, Llama-3-Korean-Bllossom-8B).

Literature: expert VP-criteria prompt + LoRA r8 + 8-bit fine-tuning -> adversarial test acc
94.64% (robust vs KoBERT's 56%). = the **direct comparison group** for MiLTL Gate-2 (the MiniLM
final adjudicator). Weights unpublished -> re-run the fine-tuning (single GPU, LoRA+8bit).

Scoring: LLM-as-classifier — answer "is this voice phishing?" with yes/no **token
log-probabilities** for a continuous [0,1] score (more robust than parsing generations, and
calibratable). With fit(finetune=True), LoRA fine-tune on the train split then score; otherwise
criteria-prompt zero-shot. Heavy dependencies (torch/transformers/peft/bitsandbytes) are lazily
imported.

  python scripts/run_baselines.py --bundle artifacts/baseline/bench_korccvi.jsonl \
    --detectors adapters.baselines.bllossom_llm:BllossomLLMDetector --out artifacts/baseline
"""
from __future__ import annotations

from typing import List, Sequence

from miltl.baseline.detector import BaselineDetector
from .hf_encoder import texts_labels

_INSTALL_HINT = ("torch/transformers(+peft/bitsandbytes for LoRA·8bit) 필요 — "
                 "pip install -r adapters/baselines/requirements.txt (DGX)")

# Expert VP adjudication criteria (representative — Korean gist of the paper's 11 criteria; see arXiv 2506.06180 for the exact original).
_CRITERIA = (
    "1) 수사·금융·공공기관 사칭 2) 계좌이체·현금전달·상품권 요구 3) 원격제어·악성앱 설치 유도 "
    "4) 개인정보·인증번호·비밀번호 요구 5) 긴급성·시간압박 조성 6) 비밀유지·타인상담 차단 "
    "7) 안전계좌·명의도용 등 공포 유발 8) 비정상 링크·전화번호 안내 9) 저금리 대환대출 미끼 "
    "10) 가족·지인 사칭 긴급송금 11) 정상 업무로 위장한 절차 요구")

_SYS = ("당신은 한국어 보이스피싱 탐지 전문가입니다. 아래 통화 전사가 보이스피싱인지 판정하세요. "
        f"판정 기준: {_CRITERIA}. 반드시 '예' 또는 '아니오' 한 단어로만 답하세요.")


class BllossomLLMDetector(BaselineDetector):
    """Small Korean LLM adjudicator (criteria prompt + yes/no log-probabilities). Optional LoRA fine-tuning."""

    name = "Bllossom-8B(LLM)"
    family = "llm"
    needs = frozenset({"text"})
    repro = "partial"                                   # weights unpublished -> fine-tuning must be re-run
    notes = ("B3 소형 LLM(Gate-2 비교군). 긍정/부정 첫-토큰 확률합 정규화 스코어(byte-BPE 견고). "
             "★제로샷은 약한 하한(포맷 미준수 가능) — 문헌 94.6%는 LoRA 파인튜닝 결과이므로 "
             "**finetune=True**(train 에 LoRA r8+8bit, 답만 마스킹 SFT)가 논문-비교 운영점. 가중치 미공개→재파인튜닝.")

    def __init__(self, model_name: str = "MLP-KTLim/llama-3-Korean-Bllossom-8B",
                 finetune: bool = False, load_8bit: bool = True, max_len: int = 1024,
                 epochs: int = 2, lr: float = 2e-4, batch: int = 4, seed: int = 20260705):
        self.model_name = model_name
        self.finetune, self.load_8bit = finetune, load_8bit
        self.max_len, self.epochs, self.lr, self.batch, self.seed = max_len, epochs, lr, batch, seed
        self._torch = self._tok = self._model = None
        self._yes_id = self._no_id = None

    # ---- Lazy backend ----
    def _ensure_backend(self):
        if self._torch is not None:
            return
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError as e:
            raise RuntimeError(_INSTALL_HINT) from e
        self._torch = torch
        self._AutoTok, self._AutoLM = AutoTokenizer, AutoModelForCausalLM

    def _build(self):
        torch = self._torch
        self._tok = self._AutoTok.from_pretrained(self.model_name)
        kw = {"device_map": "auto"}
        if self.load_8bit:
            try:
                from transformers import BitsAndBytesConfig
                # fp32_cpu_offload: allow parts of the 8-bit model to spill to CPU when it does not fully fit on GPU
                # (otherwise "Some modules are dispatched on the CPU or the disk" ValueError).
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
            except ImportError:
                pass                                    # full precision if bitsandbytes is absent
        else:
            kw["torch_dtype"] = torch.float16
        self._model = self._AutoLM.from_pretrained(self.model_name, **kw)
        # **Sets** of positive/negative first-token ids. Llama-3 byte-BPE encodes standalone "예" as byte
        # fragments and the token varies with context (leading space) -> collect first tokens of several
        # surface forms and sum their probabilities (robust, non-inverting).
        self._pos_ids = self._first_ids(["예", " 예", "네", " 네", "맞", " 맞", "Yes", " Yes"])
        self._neg_ids = self._first_ids(["아니", " 아니", "아니오", " 아니오", "아뇨", " 아뇨", "No", " No"])
        overlap = set(self._pos_ids) & set(self._neg_ids)      # drop ambiguous tokens (invalid if shared by both sides)
        self._pos_ids = [i for i in self._pos_ids if i not in overlap]
        self._neg_ids = [i for i in self._neg_ids if i not in overlap]

    def _first_ids(self, surfaces):
        ids = set()
        for s in surfaces:
            enc = self._tok.encode(s, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        return sorted(ids)

    def _prompt(self, transcript: str) -> str:
        msgs = [{"role": "system", "content": _SYS},
                {"role": "user", "content": f"[통화 전사]\n{transcript[:self.max_len]}\n\n판정(예/아니오):"}]
        try:
            return self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:  # noqa: BLE001 — plain-text fallback if no chat template
            return f"{_SYS}\n\n[통화 전사]\n{transcript[:self.max_len]}\n\n판정(예/아니오):"

    # ---- BaselineDetector ----
    def fit(self, train_calls: Sequence) -> None:
        self._ensure_backend()
        self._build()
        if not self.finetune:
            return                                      # zero-shot — criteria prompt only
        texts, labels = texts_labels(train_calls)
        if not texts or sum(labels) in (0, len(labels)):
            return
        torch = self._torch
        from peft import LoraConfig, get_peft_model
        self._model = get_peft_model(self._model, LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"]))
        opt = torch.optim.AdamW([p for p in self._model.parameters() if p.requires_grad], lr=self.lr)
        self._model.train()
        rng = __import__("random").Random(self.seed)
        idx = list(range(len(texts)))
        for _ in range(self.epochs):
            rng.shuffle(idx)
            for i in idx:                               # supervise only the answer tokens (yes/no) — prompt-masked SFT
                ans = " 예" if labels[i] else " 아니오"
                prompt = self._prompt(texts[i])
                enc = self._tok(prompt + ans, return_tensors="pt", truncation=True,
                                max_length=self.max_len).to(self._model.device)
                plen = len(self._tok(prompt, truncation=True, max_length=self.max_len)["input_ids"])
                lab = enc["input_ids"].clone()
                lab[0, :plen] = -100                    # exclude prompt tokens from loss -> learn only the answer
                out = self._model(**enc, labels=lab)
                out.loss.backward(); opt.step(); opt.zero_grad()

    def score(self, call) -> float:
        if self._model is None:
            return 0.0
        torch = self._torch
        self._model.eval()
        with torch.no_grad():
            enc = self._tok(self._prompt(call.transcript), return_tensors="pt",
                            truncation=True, max_length=self.max_len).to(self._model.device)
            probs = torch.softmax(self._model(**enc).logits[0, -1].float(), dim=-1)  # next-token distribution
            pos = probs[self._pos_ids].sum()
            neg = probs[self._neg_ids].sum()
            denom = (pos + neg).item()
            p = (pos.item() / denom) if denom > 1e-9 else 0.5   # positive/negative normalization -> P(phishing)
        return float(p)
