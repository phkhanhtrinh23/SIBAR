"""Packet policies: Sibar and its One-Step / Reply-Aware configurations, the floor, and the five
prior methods adapted to fact packets (Sec. 5.1, Appendix B)."""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .agent import UsabilityHead
from .cache import AgentCache
from .catalog import Catalog
from .config import SibarConfig
from .freshness import joint_freshness, local_value
from .llm import ChatBackend, extract_json
from .planner import PlanProblem
from .pool import build_pool, focal_entities
from .roles import resolve_need
from .sampler import TrajectorySampler, answer_attrs, label_key
from .types import SEARCH, AbstractNeed, Context, FactKey, Need, Session


@dataclass
class ReplyContext:
    session: Session
    i: int                       # reply turn t
    ctx: Context                 # focal entities after the reply
    answer: tuple[FactKey, ...]  # facts stated in a_t
    cache: AgentCache            # the agent's cache before P_t (digest delta_t)
    rng: np.random.Generator

    @property
    def t(self) -> int:
        return self.i


class Policy:
    name = "policy"

    def plan(self, rc: ReplyContext) -> list[FactKey]:
        raise NotImplementedError


class ReplyOnly(Policy):
    name = "Reply-Only"

    def plan(self, rc: ReplyContext) -> list[FactKey]:
        return []


# ----------------------------------------------------------------------------
class Sibar(Policy):
    """Algorithm 1, lines 1-5: pool (C1), scenarios (C2), exact selection (C3)."""

    def __init__(self, name: str, cfg: SibarConfig, catalog: Catalog, sampler: TrajectorySampler,
                 usability: UsabilityHead):
        self.name = name
        self.cfg = cfg
        self.catalog = catalog
        self.sampler = sampler
        self.usability = usability
        self.latencies: list[float] = []

    def resolve(self, scenarios, ctx: Context):
        out = []
        for sc in scenarios:
            items = []
            for it in sc.items:
                if isinstance(it, AbstractNeed):
                    items.append(resolve_need(it, ctx, self.catalog) or "UNRESOLVED")
                else:
                    items.append(it)
            out.append((items, sc.weight))
        return out

    def problem(self, rc: ReplyContext, scenarios=None) -> PlanProblem:
        cfg, cat = self.cfg, self.catalog
        H, t = cfg.horizon, rc.t
        cache = rc.cache if cfg.use_digest else None
        pool = build_pool(cat, rc.ctx, rc.answer, cache, t, H, cfg.pool_cap)
        if scenarios is None:
            scenarios = self.sampler.scenarios(rc.session, rc.i, H, rc.rng)
        answer = set(rc.answer)

        def cached_age(key: FactKey, tt: int) -> Optional[int]:
            ages = []
            if cache is not None:
                a = cache.age(key, tt)
                if a is not None:
                    ages.append(a)
            if key in answer and tt - t <= rc.cache.lease(key):
                ages.append(tt - t)
            return min(ages) if ages else None

        rate = cat.rate if cfg.lease_aware else (lambda k: 0.0)   # lease-blind: copies never go stale
        return PlanProblem(cfg, t, pool, cat.cost, rate, rc.cache.lease, cached_age,
                           self.resolve(scenarios, rc.ctx), self.usability, cfg.credit, H, cfg.gamma)

    def plan(self, rc: ReplyContext) -> list[FactKey]:
        t0 = time.perf_counter()
        res = self.problem(rc).solve()
        self.latencies.append(time.perf_counter() - t0)
        return res.packet


# ----------------------------------------------------------------------------
class PacketBaseline(Policy):
    """Prior methods build candidates without Sibar's cache-aware pool, so a packet can repeat
    facts the agent already holds. Each sends a packet only if the sum, over covered needs, of its
    own need probability times v_A exceeds lambda c(P) + C_pred."""

    def __init__(self, name: str, cfg: SibarConfig, catalog: Catalog, usability: UsabilityHead):
        self.name = name
        self.cfg = cfg
        self.catalog = catalog
        self.usability = usability

    def predict(self, rc: ReplyContext) -> list[tuple[Need, float]]:
        raise NotImplementedError

    def facts(self, rc: ReplyContext, preds: list[tuple[Need, float]]) -> list[FactKey]:
        out = []
        for need, _ in preds:
            out.extend(sorted(need.requirement))
        return list(dict.fromkeys(out))

    def _fit_budget(self, keys: list[FactKey]) -> list[FactKey]:
        packet, used = [], 0
        for k in keys:
            c = self.catalog.cost(k)
            if used + c <= self.cfg.budget:
                packet.append(k)
                used += c
        return packet

    def gate(self, rc: ReplyContext, packet: list[FactKey], preds: list[tuple[Need, float]]) -> bool:
        if not packet:
            return False
        P, t1 = set(packet), rc.t + 1
        gain = 0.0
        for need, p in preds:
            ages = []
            for k in need.requirement:
                a = rc.cache.age(k, t1)
                if k in P or k in rc.answer:
                    a = 1 if a is None else min(a, 1)
                ages.append((a, self.catalog.rate(k)))
            phi = joint_freshness(ages)
            if phi == 0:
                continue
            v = local_value(self.usability(need), phi, self.cfg.costs)
            if v >= 0:
                gain += p * v
        cost = self.cfg.costs.lam * sum(self.catalog.cost(k) for k in packet) + self.cfg.costs.c_pred
        return gain > cost

    def plan(self, rc: ReplyContext) -> list[FactKey]:
        preds = self.predict(rc)
        packet = self._fit_budget(self.facts(rc, preds))
        return packet if self.gate(rc, packet, preds) else []


def _resolved(dist, ctx: Context, catalog: Catalog, k: Optional[int] = None) -> list[tuple[Need, float]]:
    out = []
    for item, p in dist:
        if isinstance(item, AbstractNeed):
            n = resolve_need(item, ctx, catalog)
            if n is not None:
                out.append((n, p))
        if k is not None and len(out) >= k:
            break
    return out


class CompactRecord(PacketBaseline):
    """[32] a compact record (price, rating, hours, address, phone) of the slate's entities."""

    def __init__(self, *a, chain=None, **kw):
        super().__init__(*a, **kw)
        self.chain = chain

    def predict(self, rc):
        prev, cond = label_key(rc.session.label(rc.i)), answer_attrs(rc.session, rc.i)
        return _resolved(self.chain.next_distribution(prev, cond), rc.ctx, self.catalog)

    def facts(self, rc, preds):
        out = []
        for e in focal_entities(rc.ctx, rc.answer):
            for a in self.catalog.compact_attrs.get(self.catalog.domain(e), []):
                if self.catalog.has((e, a)):
                    out.append((e, a))
        return out


class CFQPPkg(PacketBaseline):
    """[10] facts of the top two questions predicted from the same S neighbors."""

    def __init__(self, *a, sampler: TrajectorySampler = None, top: int = 2, **kw):
        super().__init__(*a, **kw)
        self.sampler, self.top = sampler, top

    def predict(self, rc):
        dist = self.sampler.next_need_distribution(rc.session, rc.i, rc.rng)
        return _resolved(dist, rc.ctx, self.catalog, self.top)


class SpecAction(PacketBaseline):
    """[34, 39] the whole result of the likeliest next lookup (its attributes for the slate)."""

    def __init__(self, *a, chain=None, **kw):
        super().__init__(*a, **kw)
        self.chain = chain

    def predict(self, rc):
        prev, cond = label_key(rc.session.label(rc.i)), answer_attrs(rc.session, rc.i)
        return _resolved(self.chain.next_distribution(prev, cond), rc.ctx, self.catalog, 1)

    def facts(self, rc, preds):
        out = []
        for need, _ in preds:
            out.extend(sorted(need.requirement))
            for e in rc.ctx.slates.get(need.domain, ()):
                out.extend((e, a) for a in need.attrs if self.catalog.has((e, a)))
        return list(dict.fromkeys(out))


class TraceBundle(PacketBaseline):
    """[33] facts of the most frequent request chain (length <= H) after this kind of reply."""

    def __init__(self, *a, train: list[Session] = (), horizon: int = 6, **kw):
        super().__init__(*a, **kw)
        self.horizon = horizon
        self.chains: dict[str, Counter] = defaultdict(Counter)
        self.objs: dict[tuple, tuple] = {}
        for s in train:
            for i in range(len(s.turns)):
                chain = []
                for item in s.continuation(i, horizon):
                    if not isinstance(item, AbstractNeed):
                        break
                    chain.append(item)
                if chain:
                    key = tuple(x.label() for x in chain)
                    self.chains[label_key(s.label(i))][key] += 1
                    self.objs[key] = tuple(chain)

    def predict(self, rc):
        c = self.chains.get(label_key(rc.session.label(rc.i)))
        if not c:
            return []
        key, cnt = c.most_common(1)[0]
        p = cnt / sum(c.values())
        out = []
        for an in self.objs[key]:
            n = resolve_need(an, rc.ctx, self.catalog)
            if n is not None:
                out.append((n, p))
        return out


PROACT_SYSTEM = (
    "You are a proactive assistant that anticipates an owner's next follow-up questions about "
    "recommended items. Predict up to three information needs the owner will ask next. Reply with "
    "JSON only: {\"needs\": [{\"entities\": [<entity ids>], \"attributes\": [<attribute names>], "
    "\"confidence\": <0..1>}]}. Use only the given entity ids and attribute names."
)


class ProActPkg(PacketBaseline):
    """[16] facts of up to three LLM-predicted needs with confidence at least 0.6."""

    FALLBACK_MIN_P = 0.15

    def __init__(self, *a, predictor: Optional[ChatBackend] = None, chain=None, min_conf: float = 0.6, **kw):
        super().__init__(*a, **kw)
        self.predictor, self.chain, self.min_conf = predictor, chain, min_conf

    def predict(self, rc):
        if self.predictor is None:  # no LLM configured: label-frequency stand-in, flagged in the logs
            prev, cond = label_key(rc.session.label(rc.i)), answer_attrs(rc.session, rc.i)
            dist = self.chain.next_distribution(prev, cond)
            return [(n, p) for n, p in _resolved(dist, rc.ctx, self.catalog, 3) if p >= self.FALLBACK_MIN_P]
        cat = self.catalog
        ents = focal_entities(rc.ctx, rc.answer)[:6]
        hist = "\n".join(f"- ({t.kind}) {t.text}" for t in rc.session.turns[max(0, rc.i - 5): rc.i + 1])
        lines = [f"{k}: {cat.name(e)} ({cat.domain(e)}); attributes: {', '.join(cat.attributes(e)[:25])}"
                 for k, e in enumerate(ents)]
        user = f"Dialogue so far:\n{hist}\n\nRecommended entities:\n" + "\n".join(lines)
        out = extract_json(self.predictor.complete(PROACT_SYSTEM, user)) or {}
        preds = []
        for n in (out.get("needs", []) if isinstance(out, dict) else [])[:3]:
            try:
                conf = float(n.get("confidence", 0))
                es = tuple(ents[int(x)] for x in n.get("entities", []))
                attrs = tuple(a for a in n.get("attributes", []) if all(cat.has((e, a)) for e in es))
            except (ValueError, IndexError, TypeError, AttributeError):
                continue
            if conf >= self.min_conf and es and attrs:
                preds.append((Need(cat.domain(es[0]), "compare" if len(es) > 1 else "lookup", attrs, es), conf))
        return preds
