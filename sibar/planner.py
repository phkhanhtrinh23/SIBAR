"""C3: receding-horizon multi-step planner (Sec. 4.3).

Horizon value (Eq. 5):
    V_H(P) = sum_j w_j sum_{h<=H} gamma^{h-1} v_A(n_{j,h}; P) Serve_{t+h}(n_{j,h}, P)
             - lambda c(P) - C_pred I[P != {}]
Reply-aware value V_H^ra keeps a term only if no search turn or unserved need precedes it.
The exact selector is a branch and bound over the candidate facts with exact leaf values and
the linear relaxation of Eq. 6 (Appendix A) as node bound.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.optimize import linprog

from .config import SibarConfig
from .freshness import local_value
from .types import SEARCH, FactKey, Need

EPS = 1e-12


@dataclass
class TermFact:
    rho: float
    cached_age: Optional[int]   # age of a valid cached (or answer) copy at t+h, or None
    bit: int                    # bit of the candidate fact, 0 if not a candidate
    cand_age: Optional[int]     # h if a delivered copy is lease-valid at t+h, else None


@dataclass
class Term:
    weight: float               # w_j * gamma^{h-1} (summed over identical terms)
    u: float
    facts: list[TermFact]
    scenario: int = -1
    h: int = 0
    static: bool = True
    need_bits: int = 0          # candidates that must be delivered (no valid cached copy)

    def value(self, mask: int, costs) -> Optional[float]:
        """v_A if Serve_{t+h}(n, P) holds, else None."""
        phi = 1.0
        for f in self.facts:
            best = f.cached_age
            if f.bit and (mask & f.bit) and f.cand_age is not None:
                best = f.cand_age if best is None else min(best, f.cand_age)
            if best is None:
                return None
            if f.rho > 0:
                phi *= (1.0 - f.rho) ** best
        v = local_value(self.u, phi, costs)
        return v if v >= 0 else None


@dataclass
class PlanResult:
    packet: list[FactKey]
    value: float
    empty_value: float
    nodes: int = 0
    lps: int = 0
    seconds: float = 0.0
    terms: list[Term] = field(default_factory=list)


class PlanProblem:
    """Scenario needs mapped to requirements, cached copies and candidate copies."""

    def __init__(self, cfg: SibarConfig, t: int, candidates: list[FactKey], cost: Callable[[FactKey], int],
                 rate: Callable[[FactKey], float], lease: Callable[[FactKey], int],
                 cached_age: Callable[[FactKey, int], Optional[int]],
                 scenario_needs: list[tuple[list, float]], usability: Callable[[Need], float],
                 credit: str, horizon: int, gamma: float, budget: Optional[int] = None):
        self.cfg = cfg
        self.costs = cfg.costs
        self.t = t
        self.credit = credit
        self.budget = cfg.budget if budget is None else budget
        self.cand = list(dict.fromkeys(candidates))
        self.bit = {k: 1 << i for i, k in enumerate(self.cand)}
        self.cost_of = np.array([cost(k) for k in self.cand], dtype=float)
        self.lease = {k: lease(k) for k in self.cand}
        self.terms: list[Term] = []
        # per scenario ordered items (for reply-aware credit): list of (h, term index | marker)
        self.sequences: list[list] = []
        merged: dict[tuple, int] = {}
        for j, (items, w) in enumerate(scenario_needs):
            seq = []
            for h, item in enumerate(items[:horizon], start=1):
                if item is None or item == SEARCH:
                    seq.append((h, item))
                    continue
                if not isinstance(item, Need):           # unresolved need
                    seq.append((h, "UNRESOLVED"))
                    continue
                facts = []
                need_bits = 0
                static = True
                for key in sorted(item.requirement):
                    ca = cached_age(key, t + h)
                    b = self.bit.get(key, 0)
                    cand_age = h if (b and h <= self.lease[key]) else None
                    rho = rate(key)
                    static &= rho == 0
                    if ca is None and b and cand_age is not None:
                        need_bits |= b
                    facts.append(TermFact(rho, ca, b if cand_age is not None else 0, cand_age))
                weight = w * gamma ** (h - 1)
                term = Term(weight, usability(item), facts, j, h, static, need_bits)
                if credit == "horizon":
                    key = (item.requirement, h, item.op)
                    if key in merged:
                        self.terms[merged[key]].weight += weight
                        continue
                    merged[key] = len(self.terms)
                self.terms.append(term)
                seq.append((h, len(self.terms) - 1))
            self.sequences.append(seq)

    # --- exact objective ---------------------------------------------------
    def packet_cost(self, mask: int) -> float:
        return float(sum(self.cost_of[i] for i in range(len(self.cand)) if mask >> i & 1))

    def value(self, mask: int) -> float:
        costs = self.costs
        total = 0.0
        if self.credit == "horizon":
            for term in self.terms:
                v = term.value(mask, costs)
                if v is not None:
                    total += term.weight * v
        else:
            for seq in self.sequences:
                for _, item in seq:
                    if not isinstance(item, int):
                        break                         # search turn, session end or unresolved need
                    v = self.terms[item].value(mask, costs)
                    if v is None:
                        break                         # an unserved need brings a reply
                    total += self.terms[item].weight * v
        if mask:
            total -= costs.lam * self.packet_cost(mask) + costs.c_pred
        return total

    def mask_to_packet(self, mask: int) -> list[FactKey]:
        return [k for i, k in enumerate(self.cand) if mask >> i & 1]

    # --- bound of Eq. 6 ----------------------------------------------------
    def _bound_terms(self):
        if self.credit == "horizon":
            return self.terms
        keep = []
        for seq in self.sequences:
            for _, item in seq:
                if not isinstance(item, int):
                    break
                keep.append(self.terms[item])
        return keep

    def bound(self, inc: int, exc: int, use_lp: bool, stats: dict) -> float:
        costs = self.costs
        avail = ~exc
        const = 0.0
        lp_terms = []
        for term in self._bound_terms_cache:
            vbar = term.value(avail, costs)            # freshest copy among cached and non-excluded
            if vbar is None or vbar <= 0:
                continue
            undecided = term.need_bits & ~inc
            if undecided == 0:
                const += term.weight * vbar
            else:
                lp_terms.append((term.weight * vbar, undecided))
        inc_cost = self.packet_cost(inc)
        base = -(costs.lam * inc_cost + costs.c_pred) if inc else 0.0
        quick = const + base + sum(v for v, _ in lp_terms)
        if not use_lp or not lp_terms:
            return quick
        stats["lps"] += 1
        return self._lp(const + base, lp_terms, inc, inc_cost, quick)

    def _lp(self, const: float, lp_terms, inc: int, inc_cost: float, quick: float) -> float:
        costs = self.costs
        used = 0
        for _, b in lp_terms:
            used |= b
        fidx = [i for i in range(len(self.cand)) if used >> i & 1]
        pos = {i: k for k, i in enumerate(fidx)}
        nf, ny = len(fidx), len(lp_terms)
        has0 = inc == 0
        nv = nf + ny + (1 if has0 else 0)
        c = np.zeros(nv)
        c[:nf] = costs.lam * self.cost_of[fidx]
        for k, (v, _) in enumerate(lp_terms):
            c[nf + k] = -v
        if has0:
            c[-1] = costs.c_pred
        rows, rhs = [], []
        for k, (_, b) in enumerate(lp_terms):
            for i in fidx:
                if b >> i & 1:
                    r = np.zeros(nv)
                    r[nf + k] = 1.0
                    r[pos[i]] = -1.0
                    rows.append(r)
                    rhs.append(0.0)
        if has0:
            for k in range(nf):
                r = np.zeros(nv)
                r[k] = 1.0
                r[-1] = -1.0
                rows.append(r)
                rhs.append(0.0)
        r = np.zeros(nv)
        r[:nf] = self.cost_of[fidx]
        rows.append(r)
        rhs.append(self.budget - inc_cost)
        res = linprog(c, A_ub=np.array(rows), b_ub=np.array(rhs), bounds=[(0.0, 1.0)] * nv, method="highs")
        if res.status != 0:
            return quick
        return const - float(res.fun)

    # --- selectors -----------------------------------------------------------
    def relevant_order(self) -> list[int]:
        """Candidate facts that some creditable term needs, by potential value per token."""
        score = np.zeros(len(self.cand))
        for term in self._bound_terms():
            v = term.value(-1, self.costs)
            if v is None or v <= 0:
                continue
            bits = term.need_bits or sum(f.bit for f in term.facts)
            for i in range(len(self.cand)):
                if bits >> i & 1:
                    score[i] += term.weight * v
        rel = [i for i in range(len(self.cand)) if score[i] > 0 and self.cost_of[i] <= self.budget]
        rel.sort(key=lambda i: -score[i] / self.cost_of[i])
        return rel

    def branch_and_bound(self, use_lp: bool = True) -> PlanResult:
        t0 = time.perf_counter()
        self._bound_terms_cache = self._bound_terms()
        order = self.relevant_order()
        stats = {"nodes": 0, "lps": 0}
        greedy = self.greedy_mask(order)
        best_mask, best_val = 0, self.value(0)
        gv = self.value(greedy)
        if gv > best_val + EPS:
            best_mask, best_val = greedy, gv
        irrelevant = 0
        for i in range(len(self.cand)):
            if i not in order:
                irrelevant |= 1 << i
        stack = [(0, 0, irrelevant, 0.0)]
        while stack:
            depth, inc, exc, cost = stack.pop()
            stats["nodes"] += 1
            if inc:
                v = self.value(inc)
                if v > best_val + EPS:
                    best_mask, best_val = inc, v
            if depth == len(order):
                continue
            if self.bound(inc, exc, use_lp, stats) <= best_val + EPS:
                continue
            i = order[depth]
            b = 1 << i
            stack.append((depth + 1, inc, exc | b, cost))
            if cost + self.cost_of[i] <= self.budget:
                stack.append((depth + 1, inc | b, exc, cost + self.cost_of[i]))
        empty = self.value(0)
        if best_val <= empty:
            best_mask, best_val = 0, empty
        return PlanResult(self.mask_to_packet(best_mask), best_val, empty, stats["nodes"], stats["lps"],
                          time.perf_counter() - t0, self.terms)

    def greedy_mask(self, order: Optional[list[int]] = None) -> int:
        """Add residual requirement sets by value per token up to the budget."""
        if order is None:
            self._bound_terms_cache = self._bound_terms()
            order = self.relevant_order()
        allowed = 0
        for i in order:
            allowed |= 1 << i
        groups = {}
        for term in self._bound_terms():
            bits = sum(f.bit for f in term.facts) & allowed
            if bits:
                groups[bits] = True
        mask, cur = 0, self.value(0)
        while True:
            best = None
            for bits in groups:
                add = bits & ~mask
                if not add:
                    continue
                cand = mask | add
                c = self.packet_cost(cand)
                if c > self.budget:
                    continue
                gain = self.value(cand) - cur
                per = gain / max(1.0, self.packet_cost(add))
                if gain > EPS and (best is None or per > best[0]):
                    best = (per, cand, cur + gain)
            if best is None:
                return mask
            _, mask, cur = best

    def greedy(self) -> PlanResult:
        t0 = time.perf_counter()
        mask = self.greedy_mask()
        v, e = self.value(mask), self.value(0)
        if v <= e:
            mask, v = 0, e
        return PlanResult(self.mask_to_packet(mask), v, e, seconds=time.perf_counter() - t0, terms=self.terms)

    def solve(self) -> PlanResult:
        if self.cfg.selector == "greedy":
            return self.greedy()
        return self.branch_and_bound()
