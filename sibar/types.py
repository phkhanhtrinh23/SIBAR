"""Data model of Section 3: facts, needs, turns, sessions and the reply-time context."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

FactKey = tuple[str, str]          # (entity id, attribute)

SEARCH = "SEARCH"                  # label of a search turn inside a scenario


class SearchLabel(str):
    """A search turn label that equals SEARCH and remembers the domain searched."""

    def __new__(cls, domain: Optional[str] = None):
        obj = super().__new__(cls, SEARCH)
        obj.domain = domain
        return obj


ROLES = ("last", "selected", "first", "second", "cheapest")


@dataclass
class Entity:
    eid: str
    domain: str
    name: str
    attrs: dict[str, str]


UNRESOLVED = "<unresolved>"     # entity of a follow-up whose need the catalog cannot state


@dataclass(frozen=True)
class Need:
    """n = (kappa_n, Omega_n): an attribute operator over an entity scope."""
    domain: str
    op: str                         # lookup | compare
    attrs: tuple[str, ...]
    entities: tuple[str, ...]

    @property
    def requirement(self) -> frozenset[FactKey]:
        return frozenset((e, a) for e in self.entities for a in self.attrs)


@dataclass(frozen=True)
class AbstractNeed:
    """A need stored by entity role, so scenarios carry no other owner's entities."""
    domain: str
    op: str
    attrs: tuple[str, ...]
    roles: tuple[str, ...]

    def label(self) -> str:
        return f"{self.domain}|{self.op}|{','.join(self.attrs)}|{','.join(self.roles)}"


ScenarioItem = Optional[object]     # SEARCH, AbstractNeed, or None (session ended)


@dataclass
class Context:
    """What the service and the agent know about the focal entities before a turn."""
    slates: dict[str, tuple[str, ...]] = field(default_factory=dict)  # latest slate per domain
    focal: tuple[str, ...] = ()                                       # entities of the last m replies
    last: dict[str, str] = field(default_factory=dict)                # last mentioned per domain
    selected: dict[str, str] = field(default_factory=dict)            # owner-selected per domain


@dataclass
class Turn:
    idx: int
    kind: str                        # "search" | "followup"
    text: str
    domain: str
    slate: tuple[str, ...] = ()      # entities recommended by the reply to a search turn
    stated: tuple[FactKey, ...] = () # facts stated by the logged answer of a search turn
    need: Optional[Need] = None      # gold need of a follow-up
    abstract: Optional[AbstractNeed] = None
    schema: tuple[str, ...] = ()     # slot schema of the prefix up to this turn
    named: tuple[str, ...] = ()      # shown entities the owner names in this turn
    requested: tuple[tuple[str, str], ...] = ()  # (domain, slot) pairs the owner requests
    reply: str = ""                  # the logged service reply to this turn

    @property
    def is_search(self) -> bool:
        return self.kind == "search"


@dataclass
class Session:
    sid: str
    seeker: str
    turns: list[Turn]
    contexts: list[Context] = field(default_factory=list)  # context *after* the reply at turn i

    @property
    def n_followups(self) -> int:
        return sum(1 for t in self.turns if t.kind == "followup")

    def last_answer(self, i: int) -> str:
        """The service reply the owner saw before turn i (the agent sees it with the question)."""
        return self.turns[i - 1].reply if i > 0 else ""

    def label(self, i: int):
        t = self.turns[i]
        return SearchLabel(t.domain) if t.is_search else t.abstract

    def continuation(self, i: int, horizon: int) -> list:
        """True next `horizon` owner turns after turn i (hindsight scenario)."""
        out = []
        for h in range(1, horizon + 1):
            j = i + h
            out.append(self.label(j) if j < len(self.turns) else None)
        return out
