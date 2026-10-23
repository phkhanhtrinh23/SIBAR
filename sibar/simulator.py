"""Whole-session replay (Sec. 5.1 Protocol and Algorithm 1, lines 6-13).

Each session is replayed turn by turn; policies differ only in which facts travel with each reply.
At a follow-up the agent answers locally if its gate holds, otherwise it calls, and that call is a
reply that carries a new packet. Later turns follow the log.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from .agent import Agent
from .cache import AgentCache
from .catalog import Catalog, ChurnProcess
from .config import SibarConfig
from .policies import Policy, ReplyContext
from .types import Session


def stable_seed(*parts) -> int:
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


@dataclass
class SessionResult:
    sid: str
    seeker: str
    policy: str
    replay: int
    followups: int = 0
    correct: int = 0
    wrong: int = 0
    calls: int = 0
    tokens: int = 0
    packets: int = 0
    replies: int = 0
    # (position of the follow-up after the last search reply, answered locally and correctly)
    positions: list[tuple[int, bool]] = field(default_factory=list)

    def J(self, cfg: SibarConfig) -> float:
        """Realized Eq. 2: +C_call per correct local answer, -C_err per wrong one, minus delivery."""
        c = cfg.costs
        # Sec. 5.3 variant: a planner pays C_pred at every reply, even one that sends nothing;
        # the answers-only floor plans nothing and pays none
        planner = self.policy != "Reply-Only"
        charged = self.replies if (c.charge_pred_every_reply and planner) else self.packets
        return c.c_call * self.correct - c.c_err * self.wrong - c.lam * self.tokens - c.c_pred * charged


def replay_session(session: Session, policy: Policy, agent: Agent, catalog: Catalog, cfg: SibarConfig,
                   replay: int = 0) -> SessionResult:
    res = SessionResult(session.sid, session.seeker, policy.name, replay)
    churn = ChurnProcess(catalog, len(session.turns) + 1,
                         np.random.default_rng(stable_seed("churn", session.sid, replay)))
    rng = np.random.default_rng(stable_seed("policy", session.sid, replay))
    cache = AgentCache(catalog, cfg.costs, cfg.lease_cap_turns, cfg.lease_aware)
    pos = 0
    for i, turn in enumerate(session.turns):
        if turn.is_search:
            pos = 0
            answer = tuple(k for k in turn.stated if catalog.has(k))
        else:
            pos += 1
            res.followups += 1
            out = agent.try_local(turn, cache, i, churn, session.last_answer(i))
            if out.local:
                res.correct += int(out.correct)
                res.wrong += int(not out.correct)
                res.positions.append((pos, out.correct))
                continue                                   # answered with zero calls
            res.calls += 1
            res.positions.append((pos, False))
            # the fallback answer states R(n); a need on facts the catalog lacks is always a call
            answer = tuple(k for k in sorted(turn.need.requirement) if catalog.has(k))
        cache.add(answer, i, churn)
        packet = policy.plan(ReplyContext(session, i, session.contexts[i], answer, cache, rng))
        cache.add(packet, i, churn)
        res.tokens += sum(catalog.cost(k) for k in packet)
        res.packets += int(bool(packet))
        res.replies += 1
    return res
