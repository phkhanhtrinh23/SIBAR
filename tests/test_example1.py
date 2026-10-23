"""Example 1 (running example) and Proposition 3(ii) of the paper."""
import pytest

from sibar.config import CostConfig, SibarConfig
from sibar.planner import PlanProblem
from sibar.types import SEARCH, Need

A, B = "hotel:A", "hotel:B"
F = {"f1": (A, "breakfast"), "f2": (B, "breakfast"), "f3": (B, "transit_fare"),
     "f4": (B, "transit_time"), "f5": (A, "late_checkout")}
COST = {F["f1"]: 18, F["f2"]: 18, F["f3"]: 24, F["f4"]: 20, F["f5"]: 18}
n1 = Need("hotel", "compare", ("breakfast",), (A, B))
n2 = Need("hotel", "lookup", ("transit_fare", "transit_time"), (B,))
n3 = Need("hotel", "lookup", ("late_checkout",), (A,))
SCENARIOS = [([n1, SEARCH, n2, None], 0.6), ([n1, n3, SEARCH, n2], 0.2), ([n1, n2, SEARCH, n3], 0.2)]
CFG = SibarConfig(budget=80, costs=CostConfig(c_call=1.0, c_err=2.0, lam=0.002, c_pred=0.04))


def problem(credit, horizon, gamma, scenarios=SCENARIOS):
    return PlanProblem(CFG, 0, list(F.values()), COST.__getitem__, lambda k: 0.0, lambda k: 1000,
                       lambda k, t: None, scenarios, lambda n: 1.0, credit, horizon, gamma)


def mask(p, names):
    return sum(p.bit[F[n]] for n in names)


def test_horizon_values():
    p = problem("horizon", 4, 0.9)
    assert p.value(mask(p, ["f1", "f2", "f3", "f4"])) == pytest.approx(1.61, abs=0.005)
    assert p.value(mask(p, ["f1", "f2", "f5"])) == pytest.approx(1.18, abs=0.005)


def test_packets_by_configuration():
    one = problem("horizon", 1, 1.0).branch_and_bound()
    assert set(one.packet) == {F["f1"], F["f2"]}
    ra = problem("reply_aware", 4, 1.0).branch_and_bound()
    assert set(ra.packet) == {F["f1"], F["f2"], F["f5"]}
    sib = problem("horizon", 4, 0.9).branch_and_bound()
    assert set(sib.packet) == {F["f1"], F["f2"], F["f3"], F["f4"]}


def test_not_submodular():
    """Prop. 3(ii): adding f2 to {} changes V_1 by -0.076, adding it to {f1} by +0.964."""
    p = problem("horizon", 1, 1.0)
    assert p.value(mask(p, ["f2"])) - p.value(0) == pytest.approx(-0.076)
    assert p.value(mask(p, ["f1", "f2"])) - p.value(mask(p, ["f1"])) == pytest.approx(0.964)


@pytest.mark.parametrize("credit,H,gamma", [("horizon", 1, 1.0), ("horizon", 4, 0.9), ("reply_aware", 4, 1.0)])
def test_branch_and_bound_is_exact(credit, H, gamma):
    """Prop. 3(iii): B&B with the LP bound matches brute-force enumeration."""
    p = problem(credit, H, gamma)
    best = max(p.value(m) for m in range(1 << len(p.cand)) if p.packet_cost(m) <= CFG.budget)
    assert p.branch_and_bound().value == pytest.approx(best)
    assert p.branch_and_bound(use_lp=False).value == pytest.approx(best)
