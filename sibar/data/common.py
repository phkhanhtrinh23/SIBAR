"""Dataset container, splits and shared post-processing."""
from __future__ import annotations

import hashlib
import os
import pickle
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..catalog import Catalog
from ..roles import build_contexts
from ..tokens import truncate_tokens
from ..types import UNRESOLVED, Need, Session


def unresolved_need(domain: str, attrs) -> Need:
    """A follow-up stays a follow-up even when its need does not resolve; it is always a call."""
    return Need(domain, "lookup", tuple(attrs) or ("?",), (UNRESOLVED,))


@dataclass
class Dataset:
    name: str
    catalog: Catalog
    train: list[Session]
    dev: list[Session]
    test: list[Session]
    usability_fold: list[Session] = field(default_factory=list)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "Dataset":
        with open(path, "rb") as f:
            return pickle.load(f)

    def stats(self) -> dict:
        def n(ss):
            return {"sessions": len(ss), "followups": sum(s.n_followups for s in ss),
                    "seekers": len({s.seeker for s in ss})}
        return {"entities": len(self.catalog.entities), "train": n(self.train), "dev": n(self.dev),
                "test": n(self.test), "usability_fold": n(self.usability_fold)}


def summary_value(items: list[str], top_k: int, max_tokens: int) -> str:
    """Relational neighbourhoods enter as one bounded summary fact each."""
    return truncate_tokens(", ".join(items[:top_k]), max_tokens)


def finalize(name: str, catalog: Catalog, train: list[Session], dev: list[Session], test: list[Session],
             usability_share: float = 0.1, focus_replies: int = 3, seed: int = 0) -> Dataset:
    for s in train + dev + test:
        build_contexts(s, catalog, focus_replies)
    # schema priority for the pool cap: how often owners ask about (domain, attribute)
    freq: Counter = Counter()
    for s in train:
        for t in s.turns:
            if t.need is not None and UNRESOLVED not in t.need.entities:
                for a in t.need.attrs:
                    freq[(t.need.domain, a)] += 1
    catalog.priority = {k: float(v) for k, v in freq.items()}
    # a separate fold of training sessions for the usability head (by seeker when possible)
    rng = np.random.default_rng(seed)
    seekers = sorted({s.seeker for s in train})
    if len(seekers) >= 10:
        k = max(1, int(round(usability_share * len(seekers))))
        fold_seekers = set(rng.choice(seekers, k, replace=False).tolist())
        fold = [s for s in train if s.seeker in fold_seekers]
        rest = [s for s in train if s.seeker not in fold_seekers]
    else:
        idx = rng.permutation(len(train))
        k = int(round(usability_share * len(train)))
        fold = [train[i] for i in idx[:k]]
        rest = [train[i] for i in idx[k:]]
    return Dataset(name, catalog, rest, dev, test, fold)


def split_by_hash(sessions: list[Session], salt: str, test_share: float = 0.2, dev_share: float = 0.2,
                  seed: int = 0):
    """Seeker-disjoint split by a salted hash of the seeker id, so it does not depend on the seed."""
    def held_out(seeker: str) -> bool:
        x = int.from_bytes(hashlib.sha256((salt + seeker).encode()).digest()[:8], "big") / 2 ** 64
        return x >= 1 - test_share
    test = [s for s in sessions if held_out(s.seeker)]
    rest = [s for s in sessions if not held_out(s.seeker)]
    seekers = sorted({s.seeker for s in rest})
    rng = np.random.default_rng(seed)
    dev_s = set(rng.choice(seekers, max(1, int(round(dev_share * len(seekers)))), replace=False).tolist())
    return [s for s in rest if s.seeker not in dev_s], [s for s in rest if s.seeker in dev_s], test


def split_by_seeker(sessions: list[Session], n_test: int, dev_share: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    seekers = sorted({s.seeker for s in sessions})
    perm = [seekers[i] for i in rng.permutation(len(seekers))]
    test_s = set(perm[:n_test])
    train_seekers = perm[n_test:]
    n_dev = max(1, int(round(dev_share * len(train_seekers))))
    dev_s = set(train_seekers[:n_dev])
    test = [s for s in sessions if s.seeker in test_s]
    dev = [s for s in sessions if s.seeker in dev_s]
    train = [s for s in sessions if s.seeker not in test_s and s.seeker not in dev_s]
    return train, dev, test
