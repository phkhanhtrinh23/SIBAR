"""The agent's cache M and its digest delta_t (Definition 1, Sec. 4.1, C4)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .catalog import Catalog, ChurnProcess
from .config import CostConfig
from .freshness import lease_turns
from .types import FactKey


@dataclass
class Copy:
    value: str
    version: int
    t: int            # delivery turn
    lease: int        # Delta(a) in turns
    sig: str


class AgentCache:
    def __init__(self, catalog: Catalog, costs: CostConfig, lease_cap: int, lease_aware: bool = True):
        self.catalog = catalog
        self.costs = costs
        self.lease_cap = lease_cap
        self.lease_aware = lease_aware
        self.copies: dict[FactKey, Copy] = {}

    def lease(self, key: FactKey) -> int:
        if not self.lease_aware:
            return self.lease_cap
        return lease_turns(self.catalog.rate(key), self.costs, self.lease_cap)

    def add(self, keys: Iterable[FactKey], t: int, churn: ChurnProcess) -> None:
        for key in keys:
            ver = churn.version(key, t)
            val = churn.value(key, t)
            self.copies[key] = Copy(val, ver, t, self.lease(key), Catalog.sign(key, val, ver))

    def age(self, key: FactKey, t: int) -> Optional[int]:
        """Age of the cached copy at turn t if it is verified and within its lease (M_t^ok)."""
        c = self.copies.get(key)
        if c is None:
            return None
        age = t - c.t
        if age > c.lease:
            return None
        if not Catalog.verify(key, c.value, c.version, c.sig):
            return None
        return age

    def valid(self, key: FactKey, t: int) -> bool:
        return self.age(key, t) is not None

    def digest(self) -> dict[FactKey, tuple[int, int]]:
        """delta_t: cached fact identifiers with delivery times and lease ends."""
        return {k: (c.t, c.t + c.lease) for k, c in self.copies.items()}
