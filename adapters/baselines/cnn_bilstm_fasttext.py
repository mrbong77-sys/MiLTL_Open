"""B2 — Attention 1D CNN-BiLSTM + FastText text baseline (docs/BASELINES.md; selfcontrol7, MDPI Math 2023).

Literature: KorCCVi v2 acc 99.32 / F1 99.31. Faithfully reproduce the architecture
(FastText 300d -> 1D CNN (local features) -> BiLSTM (context) -> additive attention
pooling -> dense 2-class), but re-measure on the same benchmark at the low-false-positive
operating point (docs/BASELINES.md).

Tokenization: the original paper uses MeCab. Use MeCab if available; otherwise fall back to
whitespace/syllable tokens (reproduction-risk footnote). Embeddings: Korean FastText
(cc.ko.300 or self-trained on the training set) — see requirements.txt and options below.
Heavy dependencies are lazily imported.

  python scripts/run_baselines.py --bundle artifacts/baseline/bench_korccvi.jsonl \
    --detectors adapters.baselines.cnn_bilstm_fasttext:CnnBiLstmFastTextDetector --out artifacts/baseline
"""
from __future__ import annotations

import re
from typing import List, Sequence

from miltl.baseline.detector import BaselineDetector
from .hf_encoder import texts_labels

_INSTALL_HINT = ("torch(+선택 fasttext/mecab) 필요 — pip install -r adapters/baselines/requirements.txt")
_WS = re.compile(r"\s+")


def _tokenize(text: str, mecab=None) -> List[str]:
    if mecab is not None:
        try:
            return mecab.morphs(text)
        except Exception:  # noqa: BLE001
            pass
    toks = _WS.sub(" ", text).strip().split(" ")
    return toks or [text[:1]]                          # guard against empty transcript


class CnnBiLstmFastTextDetector(BaselineDetector):
    """Attention 1D CNN-BiLSTM + FastText. Embeddings: (1) pretrained fasttext .bin or (2) self-trained on the training set."""

    name = "CNN-BiLSTM+FastText(text)"
    family = "text"
    needs = frozenset({"text"})
    repro = "ok"
    notes = ("B2 Attention 1D CNN-BiLSTM+FastText. 문헌 KorCCVi acc≈0.993(소량·자가보고) — 동일 split "
             "재측정. MeCab 없으면 공백토큰 폴백(재현 리스크). fasttext_bin 미지정 시 학습셋 자체학습.")

    def __init__(self, emb_dim: int = 300, max_len: int = 128, hidden: int = 128,
                 kernels=(3, 4, 5), n_filters: int = 100, epochs: int = 5, lr: float = 1e-3,
                 batch: int = 32, seed: int = 20260705, fasttext_bin: str = None, use_mecab: bool = True):
        self.emb_dim, self.max_len, self.hidden = emb_dim, max_len, hidden
        self.kernels, self.n_filters = tuple(kernels), n_filters
        self.epochs, self.lr, self.batch, self.seed = epochs, lr, batch, seed
        self.fasttext_bin, self.use_mecab = fasttext_bin, use_mecab
        self._torch = self._model = self._ft = self._mecab = None
        self._vocab = None
        self._emb_init = None
        self._frozen = False                            # True after load() -> fit ignored

    def _ensure_backend(self):
        if self._torch is not None:
            return
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise RuntimeError(_INSTALL_HINT) from e
        self._torch = torch
        if self.use_mecab:
            try:
                from konlpy.tag import Mecab
                self._mecab = Mecab()
            except Exception:  # noqa: BLE001
                self._mecab = None            # fallback (stated in footnote)
        if self.fasttext_bin:
            import fasttext
            self._ft = fasttext.load_model(self.fasttext_bin)

    def _build_model(self, vocab_size):
        import torch.nn as nn
        ft, self_ = self, self

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.emb = nn.Embedding(vocab_size, self_.emb_dim, padding_idx=0)
                if self_._emb_init is not None:
                    s.emb.weight.data.copy_(self_._torch.tensor(self_._emb_init))
                s.convs = nn.ModuleList([nn.Conv1d(self_.emb_dim, self_.n_filters, k, padding=k // 2)
                                         for k in self_.kernels])
                s.lstm = nn.LSTM(self_.n_filters * len(self_.kernels), self_.hidden,
                                 batch_first=True, bidirectional=True)
                s.att = nn.Linear(2 * self_.hidden, 1)
                s.fc = nn.Linear(2 * self_.hidden, 2)

            def forward(s, x):
                import torch
                e = s.emb(x).transpose(1, 2)                       # B,E,L
                L = e.size(-1)
                # Even kernels with symmetric padding k//2 output L+1 -> crop to input length L so kernel outputs align.
                c = torch.cat([torch.relu(cv(e))[..., :L] for cv in s.convs], dim=1).transpose(1, 2)  # B,L,F*k
                h, _ = s.lstm(c)                                   # B,L,2H
                a = torch.softmax(s.att(h).squeeze(-1), dim=1).unsqueeze(-1)  # B,L,1
                z = (h * a).sum(1)                                 # attention pooling
                return s.fc(z)

        return Net().to("cuda" if self._torch.cuda.is_available() else "cpu")

    def _vectorize(self, tokens: List[str]) -> List[int]:
        ids = [self._vocab.get(t, 1) for t in tokens[:self.max_len]]   # 1=UNK, 0=PAD
        return ids + [0] * (self.max_len - len(ids))

    def fit(self, train_calls: Sequence) -> None:
        if self._frozen:                                # frozen load — no retraining
            return
        texts, labels = texts_labels(train_calls)
        if not texts or sum(labels) in (0, len(labels)):
            return
        self._ensure_backend()
        torch = self._torch
        torch.manual_seed(self.seed)
        toks = [_tokenize(t, self._mecab) for t in texts]
        vocab = {"<pad>": 0, "<unk>": 1}
        for ts in toks:
            for w in ts:
                vocab.setdefault(w, len(vocab))
        self._vocab = vocab
        # Embedding init: pretrained if fasttext available, else None (random, learned)
        self._emb_init = None
        if self._ft is not None:
            self._emb_init = [[0.0] * self.emb_dim] + [self._ft.get_word_vector(w).tolist()
                              if w not in ("<pad>",) else [0.0] * self.emb_dim
                              for w in list(vocab)[1:]]
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._build_model(len(vocab))
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        lossf = torch.nn.CrossEntropyLoss()
        X = torch.tensor([self._vectorize(ts) for ts in toks], device=dev)
        Y = torch.tensor(labels, device=dev)
        idx = list(range(len(texts)))
        rng = __import__("random").Random(self.seed)
        self._model.train()
        for _ in range(self.epochs):
            rng.shuffle(idx)
            for s in range(0, len(idx), self.batch):
                bi = idx[s:s + self.batch]
                out = self._model(X[bi]); loss = lossf(out, Y[bi])
                loss.backward(); opt.step(); opt.zero_grad()

    def score(self, call) -> float:
        if self._model is None:
            return 0.0
        torch = self._torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.eval()
        with torch.no_grad():
            x = torch.tensor([self._vectorize(_tokenize(call.transcript, self._mecab))], device=dev)
            p = torch.softmax(self._model(x), dim=-1)[0, 1].item()
        return float(p)

    def save(self, path: str) -> None:
        """Save torch state_dict + vocab (embedding weights are included in the state_dict)."""
        import json
        import os
        os.makedirs(path, exist_ok=True)
        self._torch.save(self._model.state_dict(), os.path.join(path, "model.pt"))
        with open(os.path.join(path, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        import json
        import os
        self._ensure_backend()
        with open(os.path.join(path, "vocab.json"), encoding="utf-8") as f:
            self._vocab = json.load(f)
        self._emb_init = None                           # weights restored from state_dict
        self._model = self._build_model(len(self._vocab))
        self._model.load_state_dict(self._torch.load(os.path.join(path, "model.pt"),
                                                     map_location="cpu"))
        self._frozen = True
