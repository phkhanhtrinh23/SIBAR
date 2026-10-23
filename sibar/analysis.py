"""Follow-up run statistics (Table 4): how far beyond the next reply follow-ups arrive."""
from __future__ import annotations

from collections import Counter, defaultdict

from .sampler import answer_attrs, schema_key
from .types import Session


def _runs(sessions: list[Session]):
    """(search reply index, [follow-up turns until the next search])."""
    for s in sessions:
        i = 0
        while i < len(s.turns):
            if s.turns[i].is_search:
                j = i + 1
                fus = []
                while j < len(s.turns) and not s.turns[j].is_search:
                    fus.append(s.turns[j])
                    j += 1
                yield s, i, fus
                i = j
            else:
                i += 1


def run_statistics(train: list[Session], test: list[Session]) -> dict[str, float]:
    allsess = train + test
    fu_total = sum(s.n_followups for s in allsess)
    far = 0
    for s in allsess:
        last = None
        for t in s.turns:
            if t.is_search:
                last = t.idx
            elif last is not None and t.idx - last >= 2:
                far += 1
    first_any = first_second = 0
    first_any_te = first_second_te = 0
    ge2 = ge3 = n_search = 0
    for s, i, fus in _runs(allsess):
        n_search += 1
        ge2 += len(fus) >= 2
        ge3 += len(fus) >= 3
        if fus:
            first_any += 1
            first_second += len(fus) >= 2
    for s, i, fus in _runs(test):
        if fus:
            first_any_te += 1
            first_second_te += len(fus) >= 2
    # Coverage@4 of need 1 / 2 from the reply state: the slots of the search turn's request set
    # condition a slot-frequency model (backed off to slot popularity with weight 10), and a need is
    # covered if all its requested slots are in the top four
    if any(t.requested for s in allsess for t in s.turns):
        def key(s, i):
            return frozenset(s.turns[i].requested)

        def labels(t):
            return frozenset(t.requested)
    else:                                        # benchmarks without request sets: need labels
        def key(s, i):
            return (schema_key(s, i), answer_attrs(s, i))

        def labels(t):
            return frozenset([t.abstract.label()])
    cond = [defaultdict(Counter), defaultdict(Counter)]
    pop = Counter()
    for s, i, fus in _runs(train):
        for pos in range(min(2, len(fus))):
            cond[pos][key(s, i)].update(labels(fus[pos]))
            pop.update(labels(fus[pos]))
    total = max(sum(pop.values()), 1)
    universe = sorted(pop, key=str)
    hit, tot = [0, 0], [0, 0]
    for s, i, fus in _runs(test):
        for pos in range(min(2, len(fus))):
            c = cond[pos].get(key(s, i), Counter())
            top = set(sorted(universe, key=lambda l: (-(c[l] + 10.0 * pop[l] / total), str(l)))[:4])
            hit[pos] += labels(fus[pos]) <= top
            tot[pos] += 1
    pct = lambda a, b: 100.0 * a / max(b, 1)
    return {
        "Follow-ups two or more turns after a search reply (%)": pct(far, fu_total),
        "First follow-ups followed by a second, all (%)": pct(first_second, first_any),
        "First follow-ups followed by a second, held-out (%)": pct(first_second_te, first_any_te),
        "Search replies followed by >= 2 follow-ups (%)": pct(ge2, n_search),
        "Search replies followed by >= 3 follow-ups (%)": pct(ge3, n_search),
        "Coverage@4 of need 1, reply state (%)": pct(hit[0], tot[0]),
        "Coverage@4 of need 2, reply state (%)": pct(hit[1], tot[1]),
        "Held-out cases (need 2)": float(tot[1]),
    }
