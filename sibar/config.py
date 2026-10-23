"""Configuration with the defaults of the paper (Section 5.1 and Appendix B)."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class CostConfig:
    c_call: float = 1.0          # cost of one call to the service (~420 ms)
    c_err: float = 2.0           # extra penalty of a wrong local answer
    lam: float = 0.0012          # token weight lambda (call units per token)
    c_pred: float = 0.038        # per-packet planning cost
    charge_pred_every_reply: bool = False  # Sec. 5.3 variant: charge C_pred on every reply


@dataclass
class SibarConfig:
    # reply budget and pool
    budget: int = 128            # B, tokens per reply
    pool_cap: int = 32           # M
    focus_replies: int = 3       # m, focal entities of the last m replies
    relation_top_k: int = 3      # relational summaries keep top-3 neighbours
    relation_max_tokens: int = 32
    lease_cap_turns: int = 1000  # wall-clock cap on a lease, in turns
    # planner
    horizon: int = 6             # H
    gamma: float = 1.0           # per-turn discount
    credit: str = "horizon"      # horizon | reply_aware
    selector: str = "exact"      # exact | greedy
    # sampler
    neighbors: int = 16          # S
    tau_w: float = 0.1
    eta: float = 10.0
    use_backoff: bool = True
    scenario_source: str = "cross_user"  # cross_user | session_only | hindsight
    oracle_mix: float = 0.0      # share of scenarios replaced by the true continuation (Fig. 5c)
    # Optional variant (off as in the paper): store scenario needs by their entity role in the
    # reply-time context, so an entity first shown after a later search has no role and earns no
    # credit. Off, roles are mapped onto the current focal entities even after a predicted search,
    # which hedges against searches that do not happen (better on dev: MultiWOZ +6.2 Net).
    roles_at_reply: bool = False
    hnsw_m: int = 16
    hnsw_ef: int = 64
    encoder: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # agent / cache
    use_digest: bool = True      # ablation "w/o cache digest" sets False
    complete_evidence_gate: bool = True
    lease_aware: bool = True     # "lease-blind" variant sets False
    costs: CostConfig = field(default_factory=CostConfig)

    def replace(self, **kw: Any) -> "SibarConfig":
        costs_kw = {k: kw.pop(k) for k in list(kw) if k in CostConfig.__dataclass_fields__}
        new = dataclasses.replace(self, **kw)
        if costs_kw:
            new.costs = dataclasses.replace(self.costs, **costs_kw)
        return new

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SibarConfig":
        d = dict(d)
        costs = CostConfig(**d.pop("costs", {}))
        return cls(costs=costs, **d)

    @classmethod
    def from_yaml(cls, path: str) -> "SibarConfig":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})
