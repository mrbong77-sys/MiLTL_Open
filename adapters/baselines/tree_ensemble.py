"""B6/B7 — CatBoost / LGBM feature-based baselines (docs/BASELINES.md; KIIT 2026 comparison group).

KIIT 2026: linear attention pooling over FastText embedding sequences -> weighted-sum sentence
vector fed as input features to tree ensembles (LGBM, CatBoost). Their observation: **ML-family
models plateau as transcript length grows** (no use of accumulation) — the comparison group for
MiLTL's 2-minute accumulation thesis. Lightweight, low-compute (minimal FLOPs in the on-device
Table 5).

Features: with FastText, mean⊕max pooling over token embeddings (≈ their sentence vector);
otherwise hashed word counts (fallback, footnoted). Heavy dependencies (catboost/lightgbm/fasttext)
are lazily imported -> package import stays stdlib-safe.

  python scripts/run_baselines.py --bundle artifacts/baseline/bench_korccvi.jsonl \
    --detectors adapters.baselines.tree_ensemble:CatBoostDetector \
    --detectors adapters.baselines.tree_ensemble:LGBMDetector --out artifacts/baseline
"""
from __future__ import annotations

from typing import List, Sequence

from miltl.baseline.detector import BaselineDetector
from .hf_encoder import texts_labels
from .cnn_bilstm_fasttext import _tokenize

_INSTALL_HINT = "catboost/lightgbm(+선택 fasttext) 필요 — pip install -r adapters/baselines/requirements.txt"


class _FeatureTreeDetector(BaselineDetector):
    """FastText mean⊕max pooled (or hashing-fallback) sentence vector -> tree ensemble. Subclass sets backend."""

    family = "text"
    needs = frozenset({"text"})
    repro = "ok"
    backend = "catboost"                                # "catboost" | "lgbm"

    def __init__(self, fasttext_bin: str = None, hash_dim: int = 4096,
                 iterations: int = 300, seed: int = 20260705):
        self.fasttext_bin, self.hash_dim = fasttext_bin, hash_dim
        self.iterations, self.seed = iterations, seed
        self._ft = self._model = None
        self._frozen = False                            # True after load() -> fit ignored

    def _ensure(self):
        if self.fasttext_bin and self._ft is None:
            import fasttext
            self._ft = fasttext.load_model(self.fasttext_bin)

    def _vec(self, text: str) -> List[float]:
        toks = _tokenize(text)
        if self._ft is not None:                        # FastText mean⊕max pooling (≈ their sentence vector)
            import numpy as np
            V = np.array([self._ft.get_word_vector(w) for w in toks]) if toks \
                else np.zeros((1, self._ft.get_dimension()))
            return np.concatenate([V.mean(0), V.max(0)]).tolist()
        import zlib
        v = [0.0] * self.hash_dim                        # fallback: hashed word counts (stable hash = reproducibility)
        for w in toks:
            v[zlib.crc32(w.encode("utf-8")) % self.hash_dim] += 1.0
        return v

    def _new_model(self):
        if self.backend == "catboost":
            from catboost import CatBoostClassifier
            return CatBoostClassifier(iterations=self.iterations, random_seed=self.seed,
                                      verbose=False, loss_function="Logloss")
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=self.iterations, random_state=self.seed, verbose=-1)

    def fit(self, train_calls: Sequence) -> None:
        if self._frozen:                                # frozen load — no retraining
            return
        texts, labels = texts_labels(train_calls)
        if not texts or sum(labels) in (0, len(labels)):
            return
        try:
            self._ensure()
            X = [self._vec(t) for t in texts]
            self._model = self._new_model().fit(X, labels)
        except ImportError as e:
            raise RuntimeError(_INSTALL_HINT) from e

    def score(self, call) -> float:
        if self._model is None:
            return 0.0
        return float(self._model.predict_proba([self._vec(call.transcript)])[0][1])

    def save(self, path: str) -> None:
        import os
        import pickle
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "tree.pkl"), "wb") as f:
            pickle.dump(self._model, f)                 # catboost/lgbm/sklearn are all picklable

    def load(self, path: str) -> None:
        import os
        import pickle
        self._ensure()                                  # fasttext for _vec (if available)
        with open(os.path.join(path, "tree.pkl"), "rb") as f:
            self._model = pickle.load(f)
        self._frozen = True


class CatBoostDetector(_FeatureTreeDetector):
    name = "CatBoost(text)"
    backend = "catboost"
    notes = "B6 CatBoost+FastText 풀링 피처(KIIT 2026 대비군). ML 계열=길이 정체(누적 미활용). fasttext_bin 옵션."


class LGBMDetector(_FeatureTreeDetector):
    name = "LGBM(text)"
    backend = "lgbm"
    notes = "B7 LGBM+FastText 풀링 피처(KIIT 2026 대비군). 경량·저연산. fasttext_bin 미지정 시 해싱 폴백."
