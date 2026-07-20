"""HF encoder text-classification baseline (docs/BASELINES.md Family 2) — reproduction adapter base.

Fine-tunes Korean encoders (KoBERT/KLUE-RoBERTa/KcELECTRA, etc.) via
AutoModelForSequenceClassification: transcript -> phishing probability. Multiple encoders
share this single base (only model_name differs) -> modular.

- Heavy dependencies (torch/transformers) are **lazily imported** inside fit()/score() ->
  module import stays stdlib-safe (importable in the dev environment; actual training and
  inference run on DGX). The core harness (miltl.baseline) remains pure stdlib.

- Running on DGX:
    pip install -r adapters/baselines/requirements.txt
    python scripts/build_benchmark.py --source korccvi --csv <KorCCViD.csv> \
      --out artifacts/baseline/bench_korccvi.jsonl
    python scripts/run_baselines.py --bundle artifacts/baseline/bench_korccvi.jsonl \
      --detectors adapters.baselines.kobert:KoBertDetector --out artifacts/baseline

- Scientific stance (docs/BASELINES.md): the goal is not to reproduce the literature's
  self-reported accuracy, but to re-measure at the **same split and the same low-false-positive
  operating point (recall@FPR, ECE)** -> train the architecture faithfully, but compare under
  control on the same benchmark.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

from miltl.baseline.detector import BaselineDetector

_INSTALL_HINT = ("torch/transformers 필요 — pip install -r adapters/baselines/requirements.txt "
                 "(DGX 에서 실행; 개발환경엔 미설치)")


def texts_labels(calls: Sequence) -> Tuple[List[str], List[int]]:
    """BenchmarkCall -> (transcript, label). needs={'text'} gating guarantees transcript presence, but filter defensively."""
    T, Y = [], []
    for c in calls:
        if c.transcript:
            T.append(c.transcript); Y.append(int(c.label))
    return T, Y


class HFEncoderDetector(BaselineDetector):
    """Encoder classifier fine-tuned via AutoModelForSequenceClassification (2-class)."""

    family = "text"
    needs = frozenset({"text"})
    repro = "ok"

    # Overridden by subclasses
    model_name = "klue/roberta-base"
    name = "hf-encoder"

    def __init__(self, model_name: str = None, name: str = None,
                 max_len: int = 256, epochs: int = 3, lr: float = 2e-5,
                 batch: int = 16, seed: int = 20260705, trust_remote_code: bool = False):
        if model_name:
            self.model_name = model_name
        if name:
            self.name = name
        self.max_len, self.epochs, self.lr = max_len, epochs, lr
        self.batch, self.seed, self.trust = batch, seed, trust_remote_code
        self._tok = self._model = self._torch = None
        self._frozen = False                            # True after load() -> fit ignored (frozen reuse)

    # ---- Lazy backend ----
    def _ensure_backend(self):
        if self._torch is not None:
            return
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError as e:  # noqa: BLE001
            raise RuntimeError(_INSTALL_HINT) from e
        self._torch = torch
        self._AutoTok = AutoTokenizer
        self._AutoModel = AutoModelForSequenceClassification

    def _device(self):
        return "cuda" if self._torch.cuda.is_available() else "cpu"

    def _build(self):
        self._tok = self._AutoTok.from_pretrained(self.model_name, trust_remote_code=self.trust)
        self._model = self._AutoModel.from_pretrained(
            self.model_name, num_labels=2, trust_remote_code=self.trust).to(self._device())

    def _encode(self, texts: List[str]):
        return self._tok(texts, truncation=True, max_length=self.max_len,
                         padding=True, return_tensors="pt").to(self._device())

    # ---- BaselineDetector ----
    def fit(self, train_calls: Sequence) -> None:
        if self._frozen:                                # frozen load — no retraining (canonical bench)
            return
        texts, labels = texts_labels(train_calls)
        if not texts or sum(labels) == 0 or sum(labels) == len(labels):
            return                                      # skip training if only one class present (runner handles it)
        self._ensure_backend()
        torch = self._torch
        torch.manual_seed(self.seed)
        self._build()
        opt = torch.optim.AdamW(self._model.parameters(), lr=self.lr)
        self._model.train()
        idx = list(range(len(texts)))
        rng = __import__("random").Random(self.seed)
        for ep in range(self.epochs):
            rng.shuffle(idx)
            for s in range(0, len(idx), self.batch):
                bi = idx[s:s + self.batch]
                enc = self._encode([texts[i] for i in bi])
                y = torch.tensor([labels[i] for i in bi], device=self._device())
                out = self._model(**enc, labels=y)
                out.loss.backward(); opt.step(); opt.zero_grad()

    def score(self, call) -> float:
        if self._model is None:
            return 0.0
        torch = self._torch
        self._model.eval()
        with torch.no_grad():
            enc = self._encode([call.transcript])
            logits = self._model(**enc).logits
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
        return float(prob)

    def save(self, path: str) -> None:
        self._ensure_backend()
        self._model.save_pretrained(path); self._tok.save_pretrained(path)

    def load(self, path: str) -> None:
        """Frozen reuse — load saved fine-tuned weights (no retraining)."""
        self._ensure_backend()
        self._tok = self._AutoTok.from_pretrained(path, trust_remote_code=self.trust)
        self._model = self._AutoModel.from_pretrained(
            path, num_labels=2, trust_remote_code=self.trust).to(self._device())
        self._frozen = True
