"""Metrics of Sec. 5.1 (Avoid, Unsup, Tok, Net) and paired seeker-level bootstrap intervals."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional

import numpy as np
from scipy import stats

from .config import SibarConfig
from .simulator import SessionResult

FLOOR = "Reply-Only"


def _groups(results: Iterable[SessionResult]) -> dict[str, dict[str, list[SessionResult]]]:
    """policy -> seeker -> results."""
    g: dict = defaultdict(lambda: defaultdict(list))
    for r in results:
        g[r.policy][r.seeker].append(r)
    return g


def _sums(rs: list[SessionResult], cfg: SibarConfig) -> np.ndarray:
    return np.array([sum(r.followups for r in rs), sum(r.correct for r in rs), sum(r.wrong for r in rs),
                     sum(r.tokens for r in rs), len(rs), sum(r.J(cfg) for r in rs),
                     sum(r.calls for r in rs)], dtype=float)


def _metrics(s: np.ndarray, s0: Optional[np.ndarray], cfg: SibarConfig) -> dict[str, float]:
    F, C, W, T, N, J, calls = s
    out = {"Avoid": 100 * C / max(F, 1), "Unsup": 100 * W / max(F, 1), "Tok": T / max(N, 1),
           "Calls": calls / max(N, 1)}
    out["Net"] = 100 * (J - s0[5]) / (cfg.costs.c_call * max(F, 1)) if s0 is not None else float("nan")
    return out


def summarize(results: list[SessionResult], cfg: SibarConfig, n_boot: int = 1000, seed: int = 0,
              reference: Optional[str] = None) -> dict[str, dict]:
    """Point estimates and 95% intervals per policy; with `reference`, also paired differences."""
    g = _groups(results)
    if FLOOR not in g:
        raise ValueError("Net needs the Reply-Only floor among the policies")
    seekers = sorted(g[FLOOR])
    G = len(seekers)
    scale = stats.t.ppf(0.975, G - 1) / stats.norm.ppf(0.975) if G > 1 else 1.0
    per = {p: np.stack([_sums(g[p].get(s, []), cfg) for s in seekers]) for p in g}
    rng = np.random.default_rng(seed)
    boots = [rng.integers(0, G, G) for _ in range(n_boot)]
    out = {}
    for p, mat in per.items():
        point = _metrics(mat.sum(0), per[FLOOR].sum(0), cfg)
        samples = {k: [] for k in point}
        diffs = {k: [] for k in point}
        for idx in boots:
            m = _metrics(mat[idx].sum(0), per[FLOOR][idx].sum(0), cfg)
            for k in m:
                samples[k].append(m[k])
            if reference and reference in per and p != reference:
                mr = _metrics(per[reference][idx].sum(0), per[FLOOR][idx].sum(0), cfg)
                for k in m:
                    diffs[k].append(m[k] - mr[k])
        row = {"point": point, "ci": {}}
        for k, v in samples.items():
            lo, hi = np.percentile(v, [2.5, 97.5])
            row["ci"][k] = (point[k] - scale * (point[k] - lo), point[k] + scale * (hi - point[k]))
        if reference and diffs["Avoid"]:
            ref_point = _metrics(per[reference].sum(0), per[FLOOR].sum(0), cfg)
            row["diff"] = {}
            for k, v in diffs.items():
                d = point[k] - ref_point[k]
                lo, hi = np.percentile(v, [2.5, 97.5])
                row["diff"][k] = (d, d - scale * (d - lo), d + scale * (hi - d))
        out[p] = row
    return out


def position_avoid(results: list[SessionResult]) -> dict[str, dict[int, float]]:
    """Avoid (%) at the 1st and 2nd follow-up after a search reply (Fig. 3)."""
    acc: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in results:
        for pos, ok in r.positions:
            if pos in (1, 2):
                acc[r.policy][pos][0] += int(ok)
                acc[r.policy][pos][1] += 1
    return {p: {k: 100 * a / max(n, 1) for k, (a, n) in d.items()} for p, d in acc.items()}


def format_table(summary: dict[str, dict], order: Optional[list[str]] = None) -> str:
    order = order or list(summary)
    lines = [f"{'Method':<22}{'Avoid':>8}{'Unsup':>8}{'Tok':>8}{'Net':>8}{'Calls':>8}   Net 95% CI"]
    for p in order:
        if p not in summary:
            continue
        m, ci = summary[p]["point"], summary[p]["ci"]["Net"]
        lines.append(f"{p:<22}{m['Avoid']:>8.1f}{m['Unsup']:>8.1f}{m['Tok']:>8.0f}{m['Net']:>8.1f}"
                     f"{m['Calls']:>8.2f}   [{ci[0]:.1f}, {ci[1]:.1f}]")
    return "\n".join(lines)
