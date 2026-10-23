"""C2: cross-user trajectory sampler (Sec. 4.2).

Session-based collaborative filtering over follow-up trajectories: the S nearest training reply
states (HNSW over psi(z_t)) give scenarios omega_j of the next H owner turns, with a smoothed
first-order chain p(o_{h+1} | o_h, A(a_t)) as backoff for sparse states.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import SibarConfig
from .encoder import StateEncoder
from .roles import reply_time_continuation
from .types import SEARCH, AbstractNeed, Session

END = "END"
MAX_H = 8


def label_key(item) -> str:
    if item is None:
        return END
    if item == SEARCH:
        return SEARCH
    return item.label()


def answer_attrs(session: Session, i: int) -> tuple[str, ...]:
    """A(a_t): the attributes the answer at turn i states."""
    t = session.turns[i]
    if t.is_search:
        return tuple(sorted({a for _, a in t.stated}))
    return tuple(sorted(t.need.attrs)) if t.need else ()


def state_text(session: Session, i: int) -> str:
    """z_t: slot schema of the prefix, attributes stated in a_t, and the kind of the turn."""
    t = session.turns[i]
    return (f"{t.kind} | {t.domain} | schema: {', '.join(t.schema)} | "
            f"stated: {', '.join(answer_attrs(session, i))}")


def schema_key(session: Session, i: int) -> str:
    t = session.turns[i]
    return f"{t.kind}|{'/'.join(t.schema)}"


@dataclass
class Scenario:
    items: list            # SEARCH | AbstractNeed | None, for h = 1..H
    weight: float
    source: str = "neighbor"


def kind_domain(label: str) -> str:
    """The (kind, domain) pair of a turn label; a scenario's search label carries no domain."""
    if label in (SEARCH, END):
        return label
    return "followup|" + label.split("|", 1)[0]


class MarkovChain:
    """Smoothed first-order chain over turn labels, conditioned on A(a_t) (Appendix B): it backs
    off from the turn label to its (kind, domain) pair and then to label frequencies, keeping each
    level with weight n/(n+beta) for n observed transitions."""

    def __init__(self, beta: float = 2.0):
        self.beta = beta
        self.c3: dict[tuple, Counter] = defaultdict(Counter)
        self.c2: dict[str, Counter] = defaultdict(Counter)
        self.c1: Counter = Counter()
        self.objects: dict[str, object] = {SEARCH: SEARCH, END: None}

    def fit(self, sessions: list[Session]) -> "MarkovChain":
        for s in sessions:
            for i in range(len(s.turns)):
                prev = label_key(s.label(i))
                self.objects.setdefault(prev, s.label(i))
                nxt_item = s.label(i + 1) if i + 1 < len(s.turns) else None
                nxt = label_key(nxt_item)
                self.objects.setdefault(nxt, nxt_item)
                cond = answer_attrs(s, i)
                self.c3[(prev, cond)][nxt] += 1
                self.c2[kind_domain(prev)][nxt] += 1
                self.c1[nxt] += 1
        self._labels = list(self.c1)
        tot = sum(self.c1.values())
        self._p1 = np.array([self.c1[l] / tot for l in self._labels])
        return self

    def dist(self, prev: str, cond: tuple) -> np.ndarray:
        b = self.beta
        c2 = self.c2.get(kind_domain(prev), Counter())
        n2 = sum(c2.values())
        p2 = (np.array([c2[l] for l in self._labels]) + b * self._p1) / (n2 + b)
        c3 = self.c3.get((prev, cond), Counter())
        n3 = sum(c3.values())
        return (np.array([c3[l] for l in self._labels]) + b * p2) / (n3 + b)

    def next_distribution(self, prev: str, cond: tuple) -> list[tuple[object, float]]:
        p = self.dist(prev, cond)
        order = np.argsort(-p)
        return [(self.objects[self._labels[k]], float(p[k])) for k in order]

    def sample(self, prev: str, cond: tuple, horizon: int, rng: np.random.Generator) -> list:
        items = []
        for _ in range(horizon):
            p = self.dist(prev, cond)
            k = int(rng.choice(len(p), p=p / p.sum()))
            lab = self._labels[k]
            items.append(self.objects[lab])
            if lab == END:
                items.extend([None] * (horizon - len(items)))
                break
            prev = lab
        return items


class TrajectorySampler:
    def __init__(self, cfg: SibarConfig, encoder: StateEncoder):
        self.cfg = cfg
        self.encoder = encoder

    # --- training -----------------------------------------------------------
    def fit(self, train: list[Session], seed: int = 0, catalog=None) -> "TrajectorySampler":
        import hnswlib

        self.train = train
        self.catalog = catalog
        self.states: list[tuple[int, int]] = []       # (session index, turn index)
        texts = []
        self.schema_counts: Counter = Counter()
        for si, s in enumerate(train):
            for i in range(len(s.turns)):
                self.states.append((si, i))
                texts.append(state_text(s, i))
                self.schema_counts[schema_key(s, i)] += 1
        self.unique_texts = list(dict.fromkeys(texts))
        tid = {t: k for k, t in enumerate(self.unique_texts)}
        self.groups: list[list[int]] = [[] for _ in self.unique_texts]
        for k, t in enumerate(texts):
            self.groups[tid[t]].append(k)
        self.emb = self.encoder.encode(self.unique_texts)
        dim = self.emb.shape[1]
        self.index = hnswlib.Index(space="ip", dim=dim)
        self.index.init_index(max_elements=len(self.unique_texts), M=self.cfg.hnsw_m,
                              ef_construction=max(100, self.cfg.hnsw_ef), random_seed=seed)
        self.index.add_items(self.emb, np.arange(len(self.unique_texts)))
        self.index.set_ef(max(self.cfg.hnsw_ef, 2 * self.cfg.neighbors))
        self.chain = MarkovChain().fit(train)
        return self

    # --- neighbors ----------------------------------------------------------
    def neighbors(self, session: Session, i: int, rng: np.random.Generator) -> list[tuple[int, float]]:
        """S nearest training reply states as (state index, similarity)."""
        S = self.cfg.neighbors
        q = self.encoder.encode([state_text(session, i)])
        k = min(len(self.unique_texts), max(S, 8))
        labels, dists = self.index.knn_query(q, k=k)
        out: list[tuple[int, float]] = []
        for lab, d in zip(labels[0], dists[0]):
            members = list(self.groups[int(lab)])
            rng.shuffle(members)                       # ties broken by the replay seed
            sim = 1.0 - float(d)                       # hnswlib "ip" returns 1 - <x, y>
            out.extend((m, sim) for m in members[: S - len(out)])
            if len(out) >= S:
                break
        return out

    def continuation(self, session: Session, i: int, horizon: int) -> list:
        catalog = getattr(self, "catalog", None)
        if self.cfg.roles_at_reply and catalog is not None:
            key = (id(session), i, horizon)
            memo = self.__dict__.setdefault("_memo", {})
            if key not in memo:
                memo[key] = reply_time_continuation(session, i, horizon, catalog)
            return list(memo[key])
        return session.continuation(i, horizon)

    def neighbor_continuation(self, state: int, horizon: int) -> list:
        si, i = self.states[state]
        return self.continuation(self.train[si], i, horizon)

    # --- scenarios ----------------------------------------------------------
    def scenarios(self, session: Session, i: int, horizon: int, rng: np.random.Generator) -> list[Scenario]:
        src = self.cfg.scenario_source
        if src == "hindsight":
            return [Scenario(self.continuation(session, i, horizon), 1.0, "truth")]
        if src == "session_only":
            out = self._session_only(session, i, horizon)
        else:
            out = self._cross_user(session, i, horizon, rng)
        if self.cfg.oracle_mix > 0:
            truth = self.continuation(session, i, horizon)
            for sc in out:
                if rng.random() < self.cfg.oracle_mix:
                    sc.items, sc.source = list(truth), "truth"
        return out

    def _cross_user(self, session: Session, i: int, horizon: int, rng) -> list[Scenario]:
        cfg = self.cfg
        nb = self.neighbors(session, i, rng)
        n_z = self.schema_counts.get(schema_key(session, i), 0)
        alpha = n_z / (n_z + cfg.eta) if cfg.use_backoff else 1.0
        out = []
        if nb and alpha > 0:
            sims = np.array([s for _, s in nb])
            w = np.exp((sims - sims.max()) / cfg.tau_w)
            w = alpha * w / w.sum()
            for (state, _), wj in zip(nb, w):
                out.append(Scenario(self.neighbor_continuation(state, horizon), float(wj), "neighbor"))
        if alpha < 1 or not out:
            share = (1 - alpha) if out else 1.0
            prev, cond = label_key(session.label(i)), answer_attrs(session, i)
            for _ in range(cfg.neighbors):
                out.append(Scenario(self.chain.sample(prev, cond, horizon, rng),
                                    share / cfg.neighbors, "chain"))
        tot = sum(s.weight for s in out)
        for s in out:
            s.weight /= tot
        return out

    @staticmethod
    def _session_only(session: Session, i: int, horizon: int) -> list[Scenario]:
        """Ablation: trajectories observed earlier in the same session only."""
        out = []
        for j in range(i):
            items = [session.label(j + h) if j + h <= i else None for h in range(1, horizon + 1)]
            if any(isinstance(x, AbstractNeed) for x in items):
                out.append(Scenario(items, 1.0, "session"))
        for s in out:
            s.weight = 1.0 / len(out)
        return out

    # --- helpers for baselines and analysis ---------------------------------
    def next_need_distribution(self, session: Session, i: int, rng) -> list[tuple[AbstractNeed, float]]:
        """CFQP-style: distribution of the next follow-up over the S neighbors."""
        nb = self.neighbors(session, i, rng)
        c: Counter = Counter()
        objs = {}
        for state, _ in nb:
            nxt = self.neighbor_continuation(state, 1)[0]
            if isinstance(nxt, AbstractNeed):
                c[nxt.label()] += 1
                objs[nxt.label()] = nxt
        n = max(1, len(nb))
        return [(objs[l], k / n) for l, k in c.most_common()]
