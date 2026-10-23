"""Proposition 1 through the full replay loop, and exactness of the selector with volatile facts."""
import numpy as np
import pytest

from sibar.agent import Agent, UsabilityHead
from sibar.catalog import Catalog
from sibar.config import CostConfig, SibarConfig
from sibar.encoder import StateEncoder
from sibar.planner import PlanProblem
from sibar.policies import Sibar
from sibar.roles import build_contexts
from sibar.sampler import TrajectorySampler
from sibar.simulator import replay_session
from sibar.types import SEARCH, Entity, Need, Session, Turn


def known_run(L):
    ent = Entity("hotel:A", "hotel", "A", {f"a{i}": f"v{i}" for i in range(L)})
    cat = Catalog([ent])
    turns = [Turn(0, "search", "find a hotel", "hotel", slate=("hotel:A",))]
    for i in range(L):
        turns.append(Turn(i + 1, "followup", f"a{i}?", "hotel", need=Need("hotel", "lookup", (f"a{i}",), ("hotel:A",))))
    s = Session("s", "u", turns)
    build_contexts(s, cat)
    return s, cat


@pytest.mark.parametrize("L", [3, 4, 7, 10])
@pytest.mark.parametrize("H", [1, 2, 3, 6])
def test_calls_by_planning_horizon(L, H):
    s, cat = known_run(L)
    cfg = SibarConfig(budget=10_000, horizon=H, gamma=1.0, scenario_source="hindsight",
                      costs=CostConfig(lam=0.0012, c_pred=0.038))
    u = UsabilityHead()
    u.default = 1.0
    for credit in ("horizon", "reply_aware"):
        c = cfg.replace(credit=credit)
        pol = Sibar("x", c, cat, TrajectorySampler(c, StateEncoder("hash")), u)
        res = replay_session(s, pol, Agent(cat, c, u), cat, c)
        assert res.calls == L // (H + 1)
        assert res.correct == L - res.calls


def test_exact_with_volatile_facts_and_cache():
    rng = np.random.default_rng(0)
    for trial in range(30):
        n_f = 7
        keys = [(f"e{i % 3}", f"a{i}") for i in range(n_f)]
        cost = {k: int(rng.integers(5, 30)) for k in keys}
        rho = {k: float(rng.choice([0.0, 0.0, 0.05, 0.2])) for k in keys}
        lease = {k: (1000 if rho[k] == 0 else int(rng.integers(0, 4))) for k in keys}
        cached = {k: int(rng.integers(0, 3)) for k in keys if rng.random() < 0.25}
        scen = []
        for _ in range(4):
            items = []
            for _h in range(4):
                if rng.random() < 0.2:
                    items.append(SEARCH)
                    continue
                ks = [keys[j] for j in rng.choice(n_f, int(rng.integers(1, 3)), replace=False)]
                ents = tuple(dict.fromkeys(k[0] for k in ks))
                attrs = tuple(dict.fromkeys(k[1] for k in ks))
                items.append(Need("d", "lookup", attrs, ents))
            scen.append((items, float(rng.random())))
        cfg = SibarConfig(budget=int(rng.integers(20, 70)))

        def cached_age(k, t):
            if k not in cached:
                return None
            age = t + cached[k]
            return age if age <= lease[k] else None

        allk = {(e, a) for items, _ in scen for n in items if isinstance(n, Need) for e in n.entities for a in n.attrs}
        for credit in ("horizon", "reply_aware"):
            p = PlanProblem(cfg, 0, keys, lambda k: cost.get(k, 10), lambda k: rho.get(k, 0.0),
                            lambda k: lease.get(k, 1000), cached_age, scen, lambda n: 0.95, credit, 4, 0.9)
            best = max(p.value(m) for m in range(1 << len(p.cand)) if p.packet_cost(m) <= cfg.budget)
            assert p.branch_and_bound().value == pytest.approx(best), (trial, credit)
        assert allk  # scenarios reference facts outside the pool too (no credit)
