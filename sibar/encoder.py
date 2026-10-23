"""State encoder psi(z_t) (d = 384) with a disk cache; a hashing fallback works offline."""
from __future__ import annotations

import hashlib
import os
import pickle
from typing import Sequence

import numpy as np


class StateEncoder:
    def __init__(self, model_name: str, cache_dir: str = ".cache", device: str | None = None):
        self.model_name = model_name
        self.cache_path = os.path.join(cache_dir, "emb_" + hashlib.md5(model_name.encode()).hexdigest()[:10] + ".pkl")
        self._mem: dict[str, np.ndarray] = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                self._mem = pickle.load(f)
        self._model = None
        self.device = device
        self.dim = 384

    def _load(self):
        if self._model is None and self.model_name != "hash":
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            dim_fn = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
            self.dim = dim_fn()
        return self._model

    @staticmethod
    def hash_embed(text: str, dim: int = 384) -> np.ndarray:
        v = np.zeros(dim, dtype=np.float32)
        for tok in text.replace("|", " ").replace(",", " ").split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % dim] += 1.0 if (h >> 20) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._mem]
        if missing:
            if self.model_name == "hash":
                for t in missing:
                    self._mem[t] = self.hash_embed(t)
            else:
                model = self._load()
                embs = model.encode(list(missing), batch_size=256, normalize_embeddings=True,
                                    show_progress_bar=len(missing) > 2000)
                for t, e in zip(missing, embs):
                    self._mem[t] = np.asarray(e, dtype=np.float32)
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(self._mem, f)
        out = np.stack([self._mem[t] for t in texts]) if texts else np.zeros((0, self.dim), np.float32)
        self.dim = out.shape[1] if len(out) else self.dim
        return out
