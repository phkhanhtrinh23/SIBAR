"""Leases, joint freshness phi and the local-answer value v_A (Definition 2, Sec. 4.1)."""
from __future__ import annotations

import math
from typing import Iterable, Optional

from .config import CostConfig


def lease_turns(rho: float, costs: CostConfig, cap: int) -> int:
    """Delta(a) = max{k : (1 - rho)^k >= C_err / (C_call + C_err)}, within a cap."""
    if rho <= 0:
        return cap
    if rho >= 1:
        return 0
    thr = costs.c_err / (costs.c_call + costs.c_err)
    if thr <= 0:
        return cap
    return min(cap, int(math.floor(math.log(thr) / math.log(1.0 - rho) + 1e-12)))


def joint_freshness(ages_rhos: Iterable[tuple[Optional[int], float]]) -> float:
    """phi = prod_f (1 - rho_f)^{k_f}; 0 if any required lease-valid copy is absent."""
    phi = 1.0
    for age, rho in ages_rhos:
        if age is None:
            return 0.0
        if rho > 0:
            phi *= (1.0 - rho) ** age
    return phi


def local_value(u: float, phi: float, costs: CostConfig) -> float:
    """v_A = theta C_call - (1 - theta) C_err with theta = u_A phi."""
    theta = u * phi
    return theta * costs.c_call - (1.0 - theta) * costs.c_err
