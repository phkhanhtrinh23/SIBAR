"""VOGUE [14]: 60 fashion-shopping dialogues over 3 catalogs of 12 items, evaluated
leave-one-seeker-out. Follow-ups are hand-grounded (operator, entity scope, attributes).

The dataset is not redistributed here. Convert it (with your grounding) to this JSON format at
data/raw/vogue/vogue.json:

{
  "catalog": [{"id": "dress_03", "domain": "fashion", "name": "...", "attrs": {"price": "59", ...}}],
  "dialogues": [{"id": "d01", "seeker": "u01", "turns": [
      {"kind": "search", "text": "...", "slate": ["dress_03", "dress_07"], "stated": [["dress_03", "price"]],
       "schema": ["fashion-color"]},
      {"kind": "followup", "text": "...", "op": "compare", "entities": ["dress_03", "dress_07"],
       "attrs": ["material"], "named": ["dress_07"]}]}]
("named": shown items the owner names in that turn; "reply": the logged reply to it; both optional.)
}
"""
from __future__ import annotations

import json
import os

from ..catalog import Catalog
from ..types import Entity, Need, Session, Turn
from .common import Dataset, finalize

VOLATILE = ["price", "availability", "rating"]


def load(raw_dir: str, test_seeker: str | None = None, seed: int = 0) -> Dataset:
    """With `test_seeker`, returns one leave-one-seeker-out fold; otherwise all dialogues as test."""
    with open(os.path.join(raw_dir, "vogue", "vogue.json")) as f:
        raw = json.load(f)
    ents = [Entity(r["id"], r.get("domain", "fashion"), r.get("name", r["id"]),
                   {k: str(v) for k, v in r["attrs"].items()}) for r in raw["catalog"]]
    cat = Catalog(ents, VOLATILE, {"fashion": "price"})
    cat.compact_attrs = {"fashion": ["price", "rating", "availability"]}
    sessions = []
    for d in raw["dialogues"]:
        turns = []
        for t in d["turns"]:
            dom = cat.domain(t["entities"][0]) if t["kind"] == "followup" else t.get("domain", "fashion")
            if t["kind"] == "followup":
                need = Need(dom, t.get("op", "lookup"), tuple(t["attrs"]), tuple(t["entities"]))
                turns.append(Turn(len(turns), "followup", t.get("text", ""), dom, need=need,
                                  schema=tuple(t.get("schema", ())), named=tuple(t.get("named", ())),
                                  reply=t.get("reply", "")))
            else:
                turns.append(Turn(len(turns), "search", t.get("text", ""), dom, slate=tuple(t.get("slate", ())),
                                  stated=tuple(tuple(x) for x in t.get("stated", ())),
                                  schema=tuple(t.get("schema", ())), named=tuple(t.get("named", ())),
                                  reply=t.get("reply", "")))
        sessions.append(Session(d["id"], d["seeker"], turns))
    if test_seeker is None:
        return finalize("vogue", cat, sessions, [], sessions, usability_share=0.0, seed=seed)
    train = [s for s in sessions if s.seeker != test_seeker]
    test = [s for s in sessions if s.seeker == test_seeker]
    return finalize("vogue", cat, train, [], test, usability_share=0.0, seed=seed)
