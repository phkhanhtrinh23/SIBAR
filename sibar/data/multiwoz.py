"""MultiWOZ 2.2 [4, 41]: Cambridge venues in 7 domains, official dialogue split (1,000 test).

Bootstrap intervals resample dialogues, so every dialogue is its own seeker.
"""
from __future__ import annotations

import glob
import json
import os
import re
from typing import Optional

from ..catalog import Catalog
from ..types import Entity, Need, Session, Turn
from .common import Dataset, finalize, unresolved_need

ATTRS = {
    "hotel": ["address", "area", "internet", "parking", "phone", "postcode", "pricerange", "stars", "type"],
    "restaurant": ["address", "area", "food", "phone", "postcode", "pricerange"],
    "attraction": ["address", "area", "entrancefee", "openhours", "phone", "postcode", "pricerange", "type"],
    "train": ["arriveby", "leaveat", "duration", "price", "departure", "destination", "day"],
    "hospital": ["address", "phone", "postcode"],
    "police": ["address", "phone", "postcode"],
}
DB_KEYS = {"entrancefee": "entrance fee", "arriveby": "arriveBy", "leaveat": "leaveAt"}
VOLATILE = ["price", "pricerange", "stars", "openhours", "entrancefee"]
PRICE = {"train": "price", "hotel": "price_single"}
COMPACT = {"hotel": ["pricerange", "stars", "address", "phone"],
           "restaurant": ["pricerange", "address", "phone"],
           "attraction": ["entrancefee", "openhours", "address", "phone"],
           "train": ["price", "leaveat", "arriveby", "duration"]}
SLATE_ACTS = {"inform", "recommend", "select", "offerbook", "offerbooked", "book"}
SKIP_REQ = {"name", "ref", "choice", "trainid"}
HOSPITAL = {"name": "addenbrookes hospital", "address": "hills rd, cambridge", "postcode": "cb20qq",
            "phone": "01223245151"}


def build_catalog(db_dir: str) -> tuple[Catalog, dict]:
    ents, by_name = [], {}

    def add(dom, name, rec):
        attrs = {}
        for a in ATTRS[dom]:
            v = rec.get(DB_KEYS.get(a, a))
            if v not in (None, "?", ""):
                attrs[a] = str(v).lower()
        if dom == "hotel" and isinstance(rec.get("price"), dict) and "single" in rec["price"]:
            attrs["price_single"] = str(rec["price"]["single"])
        eid = f"{dom}:{name}"
        ents.append(Entity(eid, dom, name, attrs))
        by_name[(dom, name)] = eid
        by_name.setdefault((dom, "~" + norm_name(name)), eid)
        by_name.setdefault((dom, "*"), eid)

    for dom in ("hotel", "restaurant", "attraction"):
        with open(os.path.join(db_dir, f"{dom}_db.json")) as f:
            for rec in json.load(f):
                add(dom, rec["name"].lower(), rec)
    with open(os.path.join(db_dir, "train_db.json")) as f:
        for rec in json.load(f):
            add("train", rec["trainID"].lower(), rec)
    with open(os.path.join(db_dir, "police_db.json")) as f:
        for rec in json.load(f):
            add("police", rec["name"].lower(), {**rec, "postcode": "cb11jg"})
    add("hospital", HOSPITAL["name"], HOSPITAL)
    cat = Catalog(ents, VOLATILE, PRICE)
    cat.compact_attrs = COMPACT
    return cat, by_name


def norm_name(name: str) -> str:
    n = re.sub(r"[^a-z0-9 ]", "", str(name).lower().replace("&", "and").replace("guest house", "guesthouse"))
    n = re.sub(r"^the ", "", n.strip())
    return re.sub(r"\s+", " ", n)


def _entity(dom: str, slot: str, value: str, by_name: dict) -> Optional[str]:
    if dom in ("hospital", "police"):
        return by_name.get((dom, "*"))
    if (dom == "train" and slot == "trainid") or (dom != "train" and slot == "name"):
        v = str(value).lower()
        e = by_name.get((dom, v)) or by_name.get((dom, "~" + norm_name(v)))
        if e is None and dom != "train":
            n = norm_name(v)
            for suffix in (" guesthouse", " hotel", " restaurant"):
                if n.endswith(suffix):
                    e = by_name.get((dom, "~" + n[: -len(suffix)]))
                    if e:
                        break
        return e
    return None


def parse_dialogue(dial: dict, acts: dict, cat: Catalog, by_name: dict) -> Session:
    turns: list[Turn] = []
    shown: dict[str, list[str]] = {}
    last_slate: dict[str, tuple] = {}
    schema: list[str] = []
    raw = dial["turns"]
    for k in range(0, len(raw) - 1, 2):
        u, s = raw[k], raw[k + 1]
        ua = acts.get(u["turn_id"], {}).get("dialog_act", {})
        sa = acts.get(s["turn_id"], {}).get("dialog_act", {})
        items = [(da.split("-")[0].lower(), da.split("-")[1], slot, val)
                 for da, svs in ua.items() if not da.startswith("general") for slot, val in svs]
        items = [x for x in items if x[0] != "booking"]
        if not items:
            continue
        dom = items[0][0]
        informs = [x for x in items if x[1] in ("Inform", "Select")]
        requests = [x for x in items if x[1] == "Request"]
        all_shown = {e for v in shown.values() for e in v}
        named = []
        for d, _, slot, val in informs:
            e = _entity(d, slot, val, by_name)
            if e is not None and e in all_shown and slot in ("name", "trainid"):
                named.append(e)
        for d, _, slot, _ in informs:
            if slot not in ("name", "trainid"):
                schema.append(f"{d}-{slot}")
        schema_t = tuple(sorted(set(schema)))
        only_naming = all(x[2] in ("name", "trainid") and _entity(x[0], x[2], x[3], by_name) in all_shown
                          for x in informs)
        need = None
        req_slots = tuple(dict.fromkeys(x[2] for x in requests if x[0] == dom and x[2] not in SKIP_REQ))
        if requests and only_naming and req_slots and dom in ATTRS and len(req_slots) == len(
                [x for x in requests if x[0] == dom]):
            ents = [e for e in named if cat.domain(e) == dom] or list(last_slate.get(dom, ()))[:1]
            ents = tuple(dict.fromkeys(ents))[:3]
            if ents and all(cat.has((e, a)) for e in ents for a in req_slots):
                need = Need(dom, "compare" if len(ents) > 1 else "lookup", req_slots, ents)
        if requests and only_naming and need is None:
            need = unresolved_need(dom, dict.fromkeys(x[2] for x in requests))
        idx = len(turns)
        if need is not None:
            turns.append(Turn(idx, "followup", u["utterance"], dom, need=need, schema=schema_t, named=tuple(named),
                              reply=s["utterance"]))
            continue
        slate, stated_slots = [], []
        for da, svs in sa.items():
            d, _, act = da.lower().partition("-")
            if d != dom:
                continue
            for slot, val in svs:
                if act in SLATE_ACTS:
                    e = _entity(d, slot, val, by_name)
                    if e is not None and e not in slate:
                        slate.append(e)
                if act == "inform":
                    stated_slots.append(slot)
        slate = tuple(slate[:5])
        stated = tuple((slate[0], a) for a in dict.fromkeys(stated_slots) if len(slate) == 1 and cat.has((slate[0], a)))
        if slate:
            shown.setdefault(dom, [])
            shown[dom].extend(e for e in slate if e not in shown[dom])
            last_slate[dom] = slate
        turns.append(Turn(idx, "search", u["utterance"], dom, slate=slate, stated=stated, schema=schema_t,
                          named=tuple(named), reply=s["utterance"]))
    return Session(dial["dialogue_id"], dial["dialogue_id"], turns)


def load(raw_dir: str, seed: int = 0) -> Dataset:
    base = os.path.join(raw_dir, "multiwoz_repo")
    cat, by_name = build_catalog(os.path.join(base, "db"))
    with open(os.path.join(base, "data", "MultiWOZ_2.2", "dialog_acts.json")) as f:
        acts = json.load(f)
    splits = {}
    for split in ("train", "dev", "test"):
        sessions = []
        for path in sorted(glob.glob(os.path.join(base, "data", "MultiWOZ_2.2", split, "dialogues_*.json"))):
            with open(path) as f:
                for dial in json.load(f):
                    s = parse_dialogue(dial, acts.get(dial["dialogue_id"], {}), cat, by_name)
                    if s.turns:
                        sessions.append(s)
        splits[split] = sessions
    return finalize("multiwoz", cat, splits["train"], splits["dev"], splits["test"], seed=seed)
