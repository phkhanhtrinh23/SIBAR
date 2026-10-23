"""CrossWOZ [42]: Beijing attractions, restaurants and hotels (5 domains with metro and taxi).

The 61 official test seekers all appear in training, so all dialogues are pooled and split by
seeker (the user worker id in `sys-usr`) with a salted hash into 64 training and 18 test seekers
(575 test sessions, Sec. 5.1).
Turn labeling follows Appendix B: a user turn with inform or select acts is a search turn, unless
its only inform act names a shown entity and asks for its attributes. Needs are the requested
(domain, slot) pairs on the focal entities; a follow-up whose need does not resolve stays a
follow-up and is always a call.
"""
from __future__ import annotations

import ast
import json
import os
import zipfile
from typing import Optional

from ..catalog import Catalog
from ..types import Entity, Need, Session, Turn
from .common import Dataset, finalize, split_by_hash, summary_value, unresolved_need

DOMAINS = {"景点": "attraction", "餐馆": "restaurant", "酒店": "hotel", "地铁": "metro", "出租": "taxi"}
SCALAR = ["地址", "地铁", "电话", "门票", "游玩时间", "评分", "营业时间", "人均消费", "价格", "酒店类型"]
RELATIONAL = ["周边景点", "周边餐馆", "周边酒店"]
VOLATILE = ["价格", "人均消费", "门票", "评分", "营业时间"]
PRICE = {"hotel": "价格", "restaurant": "人均消费", "attraction": "门票"}
COMPACT = {"hotel": ["价格", "评分", "地址", "电话"],
           "restaurant": ["人均消费", "评分", "营业时间", "地址", "电话"],
           "attraction": ["门票", "评分", "游玩时间", "地址", "电话"]}
NAMING_SLOTS = {"名称", "出发地", "目的地"}
METRO_SLOTS = {"出发地附近地铁站": "出发地", "目的地附近地铁站": "目的地"}


def _parse(x):
    return ast.literal_eval(x) if isinstance(x, str) else x


def _fmt(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, float) and v == int(v):
        v = int(v)
    return str(v)


def build_catalog(db_dir: str, top_k: int = 3, max_tokens: int = 32) -> tuple[Catalog, dict[str, str]]:
    raw = {}
    for fname, dom in (("attraction_db.json", "attraction"), ("restaurant_db.json", "restaurant"),
                       ("hotel_db.json", "hotel")):
        with open(os.path.join(db_dir, fname)) as f:
            for name, rec in json.load(f):
                raw[f"{dom}:{name}"] = (dom, name, rec)
    rating = {name: rec.get("评分") or 0 for _, name, rec in raw.values()}
    facilities = sorted({x for d, _, r in raw.values() if d == "hotel" for x in r.get("酒店设施") or []})
    ents, by_name = [], {}
    for eid, (dom, name, rec) in raw.items():
        attrs = {}
        for a in SCALAR:
            v = _fmt(rec.get(a))
            if v is not None and a in rec:
                attrs[a] = v
        if rec.get("推荐菜"):
            attrs["推荐菜"] = summary_value(list(rec["推荐菜"]), top_k, max_tokens)
        for a in RELATIONAL:
            if rec.get(a):
                nb = sorted(rec[a], key=lambda n: -float(rating.get(n) or 0))
                attrs[a] = summary_value(nb, top_k, max_tokens)
        if dom == "hotel":
            have = set(rec.get("酒店设施") or [])
            for x in facilities:
                attrs[f"酒店设施-{x}"] = "是" if x in have else "否"
        ents.append(Entity(eid, dom, name, attrs))
        by_name.setdefault(name, eid)
    cat = Catalog(ents, VOLATILE, PRICE)
    cat.compact_attrs = COMPACT
    return cat, by_name


def _load_dialogues(data_dir: str) -> dict:
    out = {}
    for split in ("train", "val", "test"):
        path = os.path.join(data_dir, f"{split}.json.zip")
        with zipfile.ZipFile(path) as z:
            with z.open(f"{split}.json") as f:
                out.update(json.load(f))
    return out


def parse_dialogue(did: str, d: dict, cat: Catalog, by_name: dict[str, str]) -> Session:
    msgs = d["messages"]
    turns: list[Turn] = []
    shown: dict[str, list[str]] = {}       # entities shown so far per domain
    last_slate: dict[str, tuple] = {}
    schema: list[str] = []
    sys_named: set = set()                 # (domain, name) the system has named so far
    for k in range(0, len(msgs) - 1, 2):
        usr, sys = msgs[k], msgs[k + 1]
        if usr["role"] != "usr" or sys["role"] != "sys":
            continue
        acts = [a for a in _parse(usr["dialog_act"]) if a[0] != "General"]
        if not acts:
            continue
        sys_acts = _parse(sys["dialog_act"])
        sys_state = _parse(sys.get("sys_state") or "{}") or {}
        dom_cn = acts[0][1]
        dom = DOMAINS.get(dom_cn, dom_cn)
        informs = [a for a in acts if a[0] in ("Inform", "Select")]
        requests = [a for a in acts if a[0] == "Request"]
        all_shown = {e for v in shown.values() for e in v}
        named = {}
        for a in informs:
            eid = by_name.get(str(a[3]))
            if a[2] in NAMING_SLOTS and eid in all_shown:
                named[a[2]] = eid
        for a in informs:
            if a[2] not in NAMING_SLOTS:
                schema.append(f"{DOMAINS.get(a[1], a[1])}-{a[2]}")
        schema_t = tuple(sorted(set(schema)))
        need = None
        # Appendix B: a follow-up requests attributes and its only inform acts name an entity the
        # system already named
        only_naming = all(a[2] == "名称" and (a[1], a[3]) in sys_named for a in informs)
        if requests and only_naming:
            if dom != "taxi" and all(r[2] != "名称" for r in requests):
                need = _need(dom, requests, named, sys_state, dom_cn, last_slate, cat, by_name)
            if need is None:
                need = unresolved_need(dom, dict.fromkeys(r[2] for r in requests))
        idx = len(turns)
        req_t = tuple(sorted({(r[1], r[2]) for r in requests}))
        sys_named |= {(a[1], a[3]) for a in sys_acts if a[2] == "名称" and a[3]}
        if need is not None:
            turns.append(Turn(idx, "followup", usr["content"], need.domain, need=need, schema=schema_t,
                              named=tuple(dict.fromkeys(named.values())), requested=req_t,
                              reply=sys["content"]))
            continue
        slate = ()
        if dom_cn in sys_state:
            slate = tuple(e for n in sys_state[dom_cn].get("selectedResults", [])
                          if (e := by_name.get(n)) and cat.domain(e) == dom)
        stated = []
        if len(slate) == 1:
            for a in sys_acts:
                if a[0] == "Inform" and a[1] == dom_cn and cat.has((slate[0], a[2])):
                    stated.append((slate[0], a[2]))
        if slate:
            shown.setdefault(dom, [])
            shown[dom].extend(e for e in slate if e not in shown[dom])
            last_slate[dom] = slate
        turns.append(Turn(idx, "search", usr["content"], dom, slate=slate, stated=tuple(dict.fromkeys(stated)),
                          schema=schema_t, named=tuple(dict.fromkeys(named.values())), requested=req_t,
                          reply=sys["content"]))
    return Session(did, str(d["sys-usr"][1]), turns)


def _need(dom, requests, named, sys_state, dom_cn, last_slate, cat, by_name) -> Optional[Need]:
    if dom == "metro":
        ents, attrs = [], []
        for r in requests:
            role = METRO_SLOTS.get(r[2])
            if role is None or role not in named:
                return None
            ents.append(named[role])
        ents = list(dict.fromkeys(ents))
        if len(ents) != 1:
            return None
        e = ents[0]
        return Need(cat.domain(e), "lookup", ("地铁",), (e,)) if cat.has((e, "地铁")) else None
    attrs = tuple(dict.fromkeys(r[2] for r in requests if r[1] == dom_cn))
    if not attrs:
        return None
    ents = [e for e in named.values() if cat.domain(e) == dom]
    if not ents and dom_cn in sys_state:
        ents = [e for n in sys_state[dom_cn].get("selectedResults", []) if (e := by_name.get(n))]
        ents = [e for e in ents if e in last_slate.get(dom, ())]
    if not ents:
        ents = list(last_slate.get(dom, ()))[:1]
    ents = tuple(dict.fromkeys(ents))[:3]
    if not ents or not all(cat.has((e, a)) for e in ents for a in attrs):
        return None
    return Need(dom, "compare" if len(ents) > 1 else "lookup", attrs, ents)


SALT = "crosswoz-next-need-v1:"


def load(raw_dir: str, dev_share: float = 0.2, seed: int = 0) -> Dataset:
    base = os.path.join(raw_dir, "crosswoz_repo", "data", "crosswoz")
    cat, by_name = build_catalog(os.path.join(base, "database"))
    dialogues = _load_dialogues(base)
    sessions = [parse_dialogue(did, d, cat, by_name) for did, d in dialogues.items()]
    sessions = [s for s in sessions if s.turns]
    train, dev, test = split_by_hash(sessions, SALT, 0.2, dev_share, seed)
    return finalize("crosswoz", cat, train, dev, test, seed=seed)
