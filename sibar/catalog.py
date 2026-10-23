"""Catalog C of atomic facts f = (e, a, v, s_f), per-attribute change rates and churn."""
from __future__ import annotations

import hashlib
import hmac
from typing import Iterable, Optional

import numpy as np

from .tokens import fact_cost
from .types import Entity, FactKey

_SERVICE_KEY = b"sibar-recommendation-service"


class Catalog:
    def __init__(self, entities: Iterable[Entity], volatile_attrs: Iterable[str] = (),
                 price_attr: Optional[dict[str, str]] = None):
        self.entities: dict[str, Entity] = {e.eid: e for e in entities}
        self.volatile_attrs = set(volatile_attrs)
        self.price_attr = price_attr or {}          # domain -> attribute used by role "cheapest"
        self.rho: dict[str, float] = {}             # per-attribute per-turn change rate
        self.priority: dict[tuple[str, str], float] = {}  # schema priority for the pool cap
        self.compact_attrs: dict[str, list[str]] = {}     # domain -> price/rating/hours/address/phone
        self._cost: dict[FactKey, int] = {}

    # --- facts -----------------------------------------------------------
    def has(self, key: FactKey) -> bool:
        e = self.entities.get(key[0])
        return e is not None and key[1] in e.attrs

    def attributes(self, eid: str) -> list[str]:
        return list(self.entities[eid].attrs)

    def value(self, key: FactKey) -> str:
        return self.entities[key[0]].attrs[key[1]]

    def domain(self, eid: str) -> str:
        return self.entities[eid].domain

    def name(self, eid: str) -> str:
        return self.entities[eid].name

    def cost(self, key: FactKey) -> int:
        c = self._cost.get(key)
        if c is None:
            c = fact_cost(self.name(key[0]), key[1], self.value(key))
            self._cost[key] = c
        return c

    def rate(self, key: FactKey) -> float:
        return self.rho.get(key[1], 0.0)

    def set_churn(self, rate: float) -> None:
        """Simulated churn of Sec. 5.6: every volatile attribute changes at `rate` per turn."""
        self.rho = {a: rate for a in self.volatile_attrs} if rate > 0 else {}

    @staticmethod
    def sign(key: FactKey, value: str, version: int) -> str:
        msg = f"{key[0]}|{key[1]}|{value}|{version}".encode()
        return hmac.new(_SERVICE_KEY, msg, hashlib.sha256).hexdigest()[:16]

    @classmethod
    def verify(cls, key: FactKey, value: str, version: int, sig: str) -> bool:
        """Verifying s_f proves provenance, not freshness."""
        return hmac.compare_digest(cls.sign(key, value, version), sig)


class ChurnProcess:
    """Stationary, independent per-turn update events with rate rho_a, sampled lazily.

    A copy delivered at turn t_d is current at t' iff no event of its fact falls in (t_d, t'].
    A changed value counts as stale even if it reverts, so values carry the version.
    """

    def __init__(self, catalog: Catalog, n_turns: int, rng: np.random.Generator):
        self.catalog = catalog
        self.n_turns = n_turns
        self.rng = rng
        self._events: dict[FactKey, np.ndarray] = {}

    def _cum(self, key: FactKey) -> Optional[np.ndarray]:
        rho = self.catalog.rate(key)
        if rho <= 0:
            return None
        ev = self._events.get(key)
        if ev is None:
            ev = np.cumsum(self.rng.random(self.n_turns + 1) < rho)
            ev[0] = 0
            self._events[key] = ev
        return ev

    def version(self, key: FactKey, t: int) -> int:
        ev = self._cum(key)
        if ev is None:
            return 0
        return int(ev[min(max(t, 0), self.n_turns)])

    def value(self, key: FactKey, t: int) -> str:
        v = self.version(key, t)
        base = self.catalog.value(key)
        return base if v == 0 else f"{base} (rev {v})"
