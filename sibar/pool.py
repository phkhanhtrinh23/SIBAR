"""C1: cache-aware session pool F_t (Eq. 4)."""
from __future__ import annotations

from typing import Iterable, Optional

from .cache import AgentCache
from .catalog import Catalog
from .types import Context, FactKey


def horizon_held(cache: Optional[AgentCache], t: int, horizon: int) -> set[FactKey]:
    """delta_t^H: horizon-valid cached facts with rho_a = 0 (volatile ones stay refresh candidates)."""
    if cache is None:
        return set()
    held = set()
    for key, c in cache.copies.items():
        static = cache.catalog.rate(key) == 0 or not cache.lease_aware
        if static and cache.age(key, t) is not None and c.t + c.lease >= t + horizon:
            held.add(key)
    return held


def focal_entities(ctx: Context, answer: Iterable[FactKey]) -> list[str]:
    ents = list(dict.fromkeys(e for e, _ in answer))
    ents += [e for e in ctx.focal if e not in ents]
    return ents


def build_pool(catalog: Catalog, ctx: Context, answer: Iterable[FactKey], cache: Optional[AgentCache],
               t: int, horizon: int, cap: int) -> list[FactKey]:
    answer = set(answer)
    held = horizon_held(cache, t, horizon)
    ents = focal_entities(ctx, answer)
    cands = []
    for rank, e in enumerate(ents):
        dom = catalog.domain(e)
        for a in catalog.attributes(e):
            key = (e, a)
            if key in answer or key in held:
                continue
            cands.append((-catalog.priority.get((dom, a), 0.0), rank, key))
    cands.sort()
    return [k for _, _, k in cands[:cap]]
