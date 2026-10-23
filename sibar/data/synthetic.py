"""A small synthetic hotel/restaurant benchmark (no download), shaped like the running example:
owners search, get a slate of two items, and ask a few follow-ups that a first-order pattern
generates (breakfast, transit, late checkout, price, ...), sometimes with a search in between."""
from __future__ import annotations

import numpy as np

from ..catalog import Catalog
from ..types import Entity, Need, Session, Turn
from .common import Dataset, finalize, split_by_seeker

HOTEL_ATTRS = {"breakfast": lambda r: f"{r.integers(0, 4) * 6} USD", "transit_fare": lambda r: f"{r.integers(1, 5)} USD",
               "transit_time": lambda r: f"{r.integers(5, 40)} min", "late_checkout": lambda r: f"{r.integers(0, 3) * 15} USD",
               "price": lambda r: f"{r.integers(90, 260)} USD", "rating": lambda r: f"{r.integers(30, 50) / 10}",
               "address": lambda r: f"{r.integers(1, 300)} Main St", "phone": lambda r: f"555-{r.integers(1000, 9999)}",
               "parking": lambda r: str(r.choice(["yes", "no"])), "wifi": lambda r: str(r.choice(["free", "paid"]))}
REST_ATTRS = {"hours": lambda r: f"{r.integers(7, 12)}:00-{r.integers(20, 24)}:00", "price": lambda r: f"{r.integers(10, 60)} USD",
              "rating": lambda r: f"{r.integers(30, 50) / 10}", "address": lambda r: f"{r.integers(1, 300)} Oak St",
              "phone": lambda r: f"555-{r.integers(1000, 9999)}", "cuisine": lambda r: str(r.choice(["thai", "pizza", "noodles"]))}
# follow-up patterns: (domain, op, attrs, roles)
PATTERNS = [
    ("hotel", "compare", ("breakfast",), ("first", "second")),
    ("hotel", "lookup", ("transit_fare", "transit_time"), ("second",)),
    ("hotel", "lookup", ("late_checkout",), ("first",)),
    ("hotel", "compare", ("price",), ("first", "second")),
    ("hotel", "lookup", ("address",), ("last",)),
    ("hotel", "lookup", ("phone",), ("last",)),
    ("restaurant", "lookup", ("hours",), ("first",)),
    ("restaurant", "lookup", ("address", "phone"), ("first",)),
]
TRANS = np.array([
    # breakfast transit checkout price address phone | search end
    [0.00, 0.45, 0.15, 0.10, 0.05, 0.00, 0.20, 0.05],
    [0.00, 0.00, 0.30, 0.05, 0.05, 0.10, 0.30, 0.20],
    [0.05, 0.20, 0.00, 0.05, 0.10, 0.10, 0.30, 0.20],
    [0.40, 0.20, 0.05, 0.00, 0.05, 0.05, 0.15, 0.10],
    [0.00, 0.10, 0.05, 0.00, 0.00, 0.50, 0.15, 0.20],
    [0.00, 0.10, 0.10, 0.00, 0.10, 0.00, 0.30, 0.40],
])


def _catalog(rng, n_hotels=60, n_rest=60) -> Catalog:
    ents = []
    for k in range(n_hotels):
        ents.append(Entity(f"hotel:H{k}", "hotel", f"Hotel {k}", {a: f(rng) for a, f in HOTEL_ATTRS.items()}))
    for k in range(n_rest):
        ents.append(Entity(f"restaurant:R{k}", "restaurant", f"Restaurant {k}", {a: f(rng) for a, f in REST_ATTRS.items()}))
    cat = Catalog(ents, volatile_attrs=["price", "rating", "hours"], price_attr={"hotel": "price", "restaurant": "price"})
    cat.compact_attrs = {"hotel": ["price", "rating", "address", "phone"], "restaurant": ["price", "rating", "hours", "address", "phone"]}
    return cat


SEARCH_COL, END_COL = 6, 7


def _session(sid: str, seeker: str, cat: Catalog, rng) -> Session:
    hotels = [e for e in cat.entities if e.startswith("hotel:")]
    rests = [e for e in cat.entities if e.startswith("restaurant:")]
    slate = {"hotel": tuple(rng.choice(hotels, 2, replace=False)), "restaurant": ()}
    schema = ["hotel-area", "hotel-price"]
    turns = [Turn(0, "search", "find a hotel near the venue", "hotel", slate=slate["hotel"],
                  stated=((slate["hotel"][0], "price"),), schema=tuple(schema))]
    state = 0 if rng.random() < 0.8 else 3
    for _ in range(8):
        dom, op, attrs, roles = PATTERNS[state]
        sl = slate[dom]
        ents = tuple(dict.fromkeys(sl[0 if r in ("first", "last") else 1] for r in roles))
        turns.append(Turn(len(turns), "followup", f"what about the {' and '.join(attrs)}?", dom,
                          need=Need(dom, op, attrs, ents), schema=tuple(schema)))
        if state >= len(TRANS):                     # restaurant follow-up: back to hotels or stop
            nxt = int(rng.choice([1, 2, END_COL], p=[0.4, 0.2, 0.4]))
        else:
            nxt = int(rng.choice(TRANS.shape[1], p=TRANS[state]))
        if nxt == END_COL:
            break
        if nxt == SEARCH_COL:
            slate["restaurant"] = tuple(rng.choice(rests, 2, replace=False))
            if "restaurant-area" not in schema:
                schema.append("restaurant-area")
            turns.append(Turn(len(turns), "search", "find a restaurant near the venue", "restaurant",
                              slate=slate["restaurant"], schema=tuple(schema)))
            nxt = 6 if rng.random() < 0.6 else 7
        state = nxt
    return Session(sid, seeker, turns)


def load(n_seekers: int = 40, sessions_per_seeker: int = 25, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    cat = _catalog(rng)
    sessions = [_session(f"s{k}_{j}", f"u{k}", cat, rng) for k in range(n_seekers) for j in range(sessions_per_seeker)]
    train, dev, test = split_by_seeker(sessions, n_test=max(2, n_seekers // 5), dev_share=0.2, seed=seed)
    return finalize("synthetic", cat, train, dev, test, seed=seed)
