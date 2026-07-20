"""B3 canonical reproduction — evaluate the kufany/VP_detector_SLM fine-tuned model **with their exact method** (docs/BASELINES.md).

Reproduction, not performance improvement. Ports the method of the original notebooks
(2.fine_tuning_SLM / 3.asking_for_SLM):
  - Load the public fine-tuned weights (Herry443/Llama-8B-KNUT-…) — full model pushed after LoRA merge.
  - Their prompt (system + prompt_format3 + 11-criteria checklist, **CoT**) verbatim.
  - **Generation** -> parse "따라서 가능도는 [N]" -> N in 0..10 -> score = N/10 (continuous, threshold-free for AUROC).
  - Threshold for point metrics chosen on train (harness) or via gamma_th (notebook range 2.1~5.0).

A **different** adapter from our bllossom_llm (single-token yes/no) — that one is our
reimplementation, this one is the original reproduction. Heavy dependencies (torch/transformers)
are lazily imported. 4-bit loading saves inference memory (weights are frozen, so it does not
affect reproducibility).

  python scripts/run_baselines.py --bundle bench_T0.jsonl \
    --detectors "adapters.baselines.bllossom_repro:BllossomReproDetector(model_name='Herry443/…')"
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

from miltl.baseline.detector import BaselineDetector

_INSTALL_HINT = "torch/transformers(+accelerate/bitsandbytes) 필요 — pip install -r adapters/baselines/requirements.txt"

# Original prompt (recovered from the notebook manifest, CoT=True). The 11-criteria checklist is to be replaced with the exact original.
_SYS = ("당신은 보이스피싱 판별기 입니다. Chain-of-thoughts 방식으로 답변하세요. "
        "평가기준의 and 나 or는 논리연산 기호입니다. ")
_PROMPT = ("평가기준을 참고하여 통화 녹취록을 읽고 보이스피싱 시도 중인지 판단하세요. "
           "보이스피싱 가능도를 0부터 10사이의 정수로 반환하세요. "
           "녹취록에 나오는 참가자가 보이스피싱 시도 중인 지에 대한 판단 근거를 녹취록의 관련 내용의 요약을 "
           "포함하여 200자 이내로 간결하게 설명해주세요. 마지막에 '따라서 가능도는 [ ]' 으로 끝맺음하고 "
           "보이스피싱 가능성이 없으면 '따라서 가능도는 [0]' 이라고 표시하세요. "
           "보이스피싱 가능도만을 표시하고 다른 얘기는 하지 마세요. 일반대화라고 판단된다면 '0' 이라고 표시하세요 ")
# Exact checklist original to be obtained via re-inspection, then swapped in. Interim = bllossom_llm._CRITERIA.
from .bllossom_llm import _CRITERIA as _CHECKLIST

_POSS = re.compile(r"가능도[는은]?\s*\[?\s*(10|[0-9])\s*\]?")   # "따라서 가능도는 [N]"
_ANYINT = re.compile(r"\b(10|[0-9])\b")


class BllossomReproDetector(BaselineDetector):
    """Original B3 reproduction — public fine-tuned weights + CoT 0-10 generation + likelihood parsing."""

    name = "B3-Bllossom-repro(CoT0-10)"
    family = "llm"
    needs = frozenset({"text"})
    repro = "ok"
    notes = "오리지널 방법 이식(공개 가중치·CoT·가능도0~10). 성능개선 아님, 재현."

    def __init__(self, model_name: str, checklist: Optional[str] = None, load_4bit: bool = True,
                 max_new_tokens: int = 256, max_len: int = 3072, gamma_th: Optional[float] = None,
                 do_sample: bool = False, temperature: float = 0.95,
                 top_p: float = 0.95, top_k: int = 50):
        # Reproducibility: the canonical bench uses do_sample=True with fixed temp/top_p/top_k, measuring variance via per-run seeds.
        self.do_sample, self.temperature, self.top_p, self.top_k = do_sample, temperature, top_p, top_k
        self.model_name = model_name
        self.checklist = checklist if checklist is not None else _CHECKLIST
        self.load_4bit, self.max_new_tokens, self.max_len = load_4bit, max_new_tokens, max_len
        self.gamma_th = gamma_th
        self._torch = self._tok = self._model = None
        self._n_scored = 0          # per-call progress log (slow generation — distinguish stall vs progress)
        self._t_start = None

    def _build(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError as e:
            raise RuntimeError(_INSTALL_HINT) from e
        self._torch = torch
        import sys
        cuda_ok = torch.cuda.is_available()
        print(f"[B3repro] cuda={cuda_ok} devices={torch.cuda.device_count() if cuda_ok else 0}",
              file=sys.stderr, flush=True)
        if not cuda_ok:
            print("[B3repro] [WARN] CUDA unavailable — CPU 폴백(8B 4bit 생성은 통화당 수 분, 사실상 불가). "
                  "GPU 확인 필요(CUDA_VISIBLE_DEVICES, torch+cu 빌드).", file=sys.stderr, flush=True)
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        kw = {"device_map": "auto"}
        _bnb = False
        if self.load_4bit:
            try:
                import bitsandbytes  # noqa: F401  (bitsandbytes actually required for 4-bit loading)
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False)
                _bnb = True
            except ImportError:
                import sys
                print("[B3repro] ⚠️ bitsandbytes 없음 → 4bit 불가, bfloat16 폴백(8B≈16GB VRAM 필요). "
                      "4bit(≈5GB) 원하면 `pip install bitsandbytes>=0.46.1`.", file=sys.stderr, flush=True)
        if not _bnb:
            kw["torch_dtype"] = torch.bfloat16
        # If a LoRA adapter repo: load base (Bllossom-8B) + adapter; if a merged full model: load directly.
        base_id = None
        try:
            from peft import PeftConfig
            base_id = PeftConfig.from_pretrained(self.model_name).base_model_name_or_path
        except Exception:  # noqa: BLE001 — no peft / not an adapter -> full-model path
            base_id = None
        if base_id:
            from peft import PeftModel
            base = AutoModelForCausalLM.from_pretrained(base_id, **kw)
            self._model = PeftModel.from_pretrained(base, self.model_name)
            if self._tok.pad_token is None:
                self._tok.pad_token = self._tok.eos_token
        else:
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kw)
        self._model.eval()
        dm = getattr(self._model, "hf_device_map", None)
        on_cpu = dm and any(str(d) in ("cpu", "disk") for d in dm.values())
        print(f"[B3repro] 로드 완료 · device_map={dm if dm else 'single'}"
              + ("  ⚠️ 일부 CPU/disk — 매우 느림" if on_cpu else ""), file=sys.stderr, flush=True)

    def fit(self, train_calls: Sequence) -> None:
        self._build()                                   # only loads the public weights (no retraining)

    def _prompt(self, transcript: str) -> str:
        user = self._PROMPT_full(transcript)
        msgs = [{"role": "system", "content": _SYS},
                {"role": "user", "content": user}]
        try:
            return self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:  # noqa: BLE001
            return f"{_SYS}\n{user}\nA:"

    def _PROMPT_full(self, transcript: str) -> str:
        return f"{_PROMPT}\n[평가기준]\n{self.checklist}\n[통화 녹취록]\n{(transcript or '')[:self.max_len]}\nA:"

    def _parse_poss(self, text: str) -> float:
        """Generated text -> likelihood 0..10 -> [0,1]. Prefer '가능도는 [N]'; else the last integer."""
        m = list(_POSS.finditer(text))
        if m:
            return int(m[-1].group(1)) / 10.0
        m2 = list(_ANYINT.finditer(text))
        if m2:
            return int(m2[-1].group(1)) / 10.0
        return 0.0

    def score(self, call) -> float:
        if self._model is None:
            return 0.5
        torch = self._torch
        import sys, time
        if self._t_start is None:
            self._t_start = time.time()
            print("[B3repro] model loaded — generation start", file=sys.stderr, flush=True)
        with torch.no_grad():
            enc = self._tok(self._prompt(call.transcript), return_tensors="pt",
                            truncation=True, max_length=self.max_len).to(self._model.device)
            gkw = dict(max_new_tokens=self.max_new_tokens, do_sample=self.do_sample,
                       pad_token_id=self._tok.eos_token_id)
            if self.do_sample:                          # fixed sampling parameters (variance via per-run seed)
                gkw.update(temperature=self.temperature, top_p=self.top_p, top_k=self.top_k)
            out = self._model.generate(**enc, **gkw)
            gen = self._tok.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True)
        p = self._parse_poss(gen)
        self._n_scored += 1
        if self._n_scored <= 5:                         # diagnostics: actual generations and parsed values (identify constant/degenerate output)
            print(f"[B3repro] gen#{self._n_scored} label={call.label} p={p:.2f} | "
                  f"{gen[:220]!r}", file=sys.stderr, flush=True)
        if self._n_scored == 1:                         # right after first generation — gauge per-call cost (total = calls in the bundle)
            print(f"[B3repro] first generation done · {time.time()-self._t_start:.1f}s "
                  f"(total = number of calls in the bundle)", file=sys.stderr, flush=True)
        if self._n_scored % 10 == 0:                    # progress every 10 calls (signals it is not stalled)
            el = time.time() - self._t_start
            print(f"[B3repro] generated {self._n_scored} · {el:.0f}s · {el/self._n_scored:.1f}s/call",
                  file=sys.stderr, flush=True)
        if self.gamma_th is not None:                   # optional gamma threshold for point decisions — AUROC uses continuous p
            return 1.0 if (p * 10.0) >= self.gamma_th else 0.0
        return p
