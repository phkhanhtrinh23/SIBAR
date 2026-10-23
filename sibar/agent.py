"""C4: the agent's verified cache and fallback controller (Sec. 4.4), plus the answerers.

The agent answers a follow-up locally only from complete, lease-valid copies whose ages keep
v_A >= 0, and otherwise calls the service. Question parsing is given (gold R^(q) = R(n)).
A local answer is correct if it states every requested value and each matches the catalog.
"""
from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .cache import AgentCache
from .catalog import Catalog, ChurnProcess
from .config import SibarConfig
from .freshness import joint_freshness, local_value
from .llm import ChatBackend, extract_json
from .types import FactKey, Need, Turn

# --- value normalization for the correctness check -------------------------
_UNITS = ["元", "分", "pounds", "pound", "gbp", "£", "$", "rmb", "minutes", "minute", "stars", "star"]


def normalize(v) -> str:
    s = str(v).strip().lower()
    for u in _UNITS:
        s = s.replace(u, "")
    s = re.sub(r"[\s,，。、;；:：'\"()（）]", "", s)
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else f"{f:g}"
    except ValueError:
        return s


# --- receiver usability u_A(n) ---------------------------------------------
class UsabilityHead:
    """u_A(n) = logistic(beta_0(kappa_n) - beta_1 |R(n)|), fit on a held-out training fold."""

    def __init__(self):
        self.beta0: dict[str, float] = {}
        self.beta1 = 0.0
        self.default = 0.99

    def fit(self, records: list[tuple[str, int, bool]]) -> "UsabilityHead":
        if not records:
            return self
        ys = np.array([r[2] for r in records], dtype=float)
        if ys.min() == ys.max():                 # a perfect (or useless) agent: constant usability
            self.default = float(np.clip(ys.mean(), 0.01, 0.999))
            self.beta0 = {}
            return self
        from sklearn.linear_model import LogisticRegression

        ops = sorted({r[0] for r in records})
        X = np.array([[1.0 if r[0] == o else 0.0 for o in ops] + [-float(r[1])] for r in records])
        lr = LogisticRegression(fit_intercept=False, C=10.0, max_iter=1000).fit(X, ys)
        coef = lr.coef_[0]
        self.beta0 = {o: float(coef[k]) for k, o in enumerate(ops)}
        self.beta1 = float(coef[-1])
        self.default = float(ys.mean())
        return self

    def __call__(self, need: Need) -> float:
        if need.op not in self.beta0:
            return self.default
        z = self.beta0[need.op] - self.beta1 * len(need.requirement)
        return 1.0 / (1.0 + math.exp(-z))

    def to_dict(self) -> dict:
        return {"beta0": self.beta0, "beta1": self.beta1, "default": self.default}


# --- answerers ----------------------------------------------------------------
@dataclass
class SliceFact:
    key: FactKey
    entity: str
    attribute: str
    value: str


class OracleAnswerer:
    """Deterministic receiver: restates the cached values it is given (no LLM)."""

    def answer(self, question: str, facts: list[SliceFact], last: str = "") -> dict[FactKey, str]:
        return {f.key: f.value for f in facts}


ANSWER_SYSTEM = (
    "You are a personal assistant agent answering your owner's follow-up question about "
    "recommended items. Use ONLY the cached facts provided. Cite every value you state. "
    "Reply with JSON only: {\"answer\": \"<short answer>\", \"citations\": "
    "[{\"id\": <fact id>, \"value\": \"<value exactly as stated>\"}]}. "
    "If the facts do not answer the question, return an empty citations list."
)


LAST_CHARS = 600                         # the last answer is cut to bound the prompt


class LLMAnswerer:
    """A frozen LLM agent that answers from the slice R^(q) of its cache, citing each value."""

    def __init__(self, backend: ChatBackend):
        self.backend = backend

    def answer(self, question: str, facts: list[SliceFact], last: str = "") -> dict[FactKey, str]:
        lines = [f"[{k}] {f.entity} | {f.attribute} | {f.value}" for k, f in enumerate(facts)]
        user = (f"Owner question: {question}\nLast answer: {last[:LAST_CHARS]}\nCached facts:\n"
                + "\n".join(lines))
        out = extract_json(self.backend.complete(ANSWER_SYSTEM, user)) or {}
        cited: dict[FactKey, str] = {}
        for c in out.get("citations", []) if isinstance(out, dict) else []:
            try:
                f = facts[int(c["id"])]
            except (KeyError, ValueError, IndexError, TypeError):
                continue
            value = str(c.get("value", ""))
            if "|" in value:                     # the agent may quote the whole fact line
                value = value.rsplit("|", 1)[1]
            cited[f.key] = value.strip()
        return cited


# --- the agent ------------------------------------------------------------------
@dataclass
class LocalOutcome:
    local: bool
    correct: bool = False
    value: float = 0.0


class Agent:
    def __init__(self, catalog: Catalog, cfg: SibarConfig, usability: UsabilityHead, answerer=None):
        self.catalog = catalog
        self.cfg = cfg
        self.usability = usability
        self.answerer = answerer or OracleAnswerer()

    def serve_value(self, need: Need, cache: AgentCache, t: int) -> Optional[float]:
        """v_A(n) if the serving gate Suff * I[v_A >= 0] holds, else None."""
        req = need.requirement
        ages = {k: cache.age(k, t) for k in req}
        rate = self.catalog.rate if self.cfg.lease_aware else (lambda k: 0.0)  # lease-blind: any copy
        if self.cfg.complete_evidence_gate:
            if any(a is None for a in ages.values()):
                return None
            phi = joint_freshness((a, rate(k)) for k, a in ages.items())
        else:                                    # ablation: answer from partial evidence
            have = {k: a for k, a in ages.items() if a is not None}
            if not have:
                return None
            phi = joint_freshness((a, rate(k)) for k, a in have.items())
        v = local_value(self.usability(need), phi, self.cfg.costs)
        return v if v >= 0 else None

    def try_local(self, turn: Turn, cache: AgentCache, t: int, churn: ChurnProcess,
                  last: str = "") -> LocalOutcome:
        need = turn.need
        v = self.serve_value(need, cache, t)
        if v is None:
            return LocalOutcome(False)
        facts = [SliceFact(k, self.catalog.name(k[0]), k[1], cache.copies[k].value)
                 for k in sorted(need.requirement) if cache.age(k, t) is not None]
        cited = self.answerer.answer(turn.text, facts, last)
        correct = all(k in cited and normalize(cited[k]) == normalize(churn.value(k, t))
                      for k in need.requirement)
        return LocalOutcome(True, correct, v)


def fit_usability(sessions, catalog: Catalog, answerer, max_records: int = 400, seed: int = 0) -> UsabilityHead:
    """Run the frozen agent on held-out training follow-ups with complete, current evidence."""
    rng = np.random.default_rng(seed)
    turns = [(t, s.last_answer(i)) for s in sessions for i, t in enumerate(s.turns)
             if t.need is not None and all(catalog.has(k) for k in t.need.requirement)]
    if len(turns) > max_records:
        turns = [turns[i] for i in rng.choice(len(turns), max_records, replace=False)]
    def record(item):
        t, last = item
        facts = [SliceFact(k, catalog.name(k[0]), k[1], catalog.value(k)) for k in sorted(t.need.requirement)]
        cited = answerer.answer(t.text, facts, last)
        ok = all(k in cited and normalize(cited[k]) == normalize(catalog.value(k)) for k in t.need.requirement)
        return t.need.op, len(t.need.requirement), ok

    with ThreadPoolExecutor(max_workers=16) as pool:
        records = list(pool.map(record, turns))
    return UsabilityHead().fit(records)
