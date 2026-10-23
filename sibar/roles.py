"""Entity roles (Appendix B) and reply-time contexts.

Needs are stored as (operator, entity role) with roles {first option, second option, cheapest,
selected, last mentioned}, then mapped onto the current focal entities.
"""
from __future__ import annotations

from typing import Optional

from .catalog import Catalog
from .types import UNRESOLVED, AbstractNeed, Context, Need, Session, Turn


def _to_float(v: str) -> Optional[float]:
    num = ""
    for ch in str(v):
        if ch.isdigit() or (ch == "." and num):
            num += ch
        elif num:
            break
    try:
        return float(num) if num else None
    except ValueError:
        return None


def cheapest(catalog: Catalog, slate: tuple[str, ...], domain: str) -> Optional[str]:
    attr = catalog.price_attr.get(domain)
    if not attr or len(slate) < 2:
        return None
    priced = [(p, e) for e in slate
              if (p := _to_float(catalog.entities[e].attrs.get(attr, ""))) is not None]
    return min(priced)[1] if priced else None


def resolve_role(role: str, domain: str, ctx: Context, catalog: Catalog) -> Optional[str]:
    slate = ctx.slates.get(domain, ())
    if role == "first":
        return slate[0] if slate else None
    if role == "second":
        return slate[1] if len(slate) > 1 else None
    if role == "cheapest":
        return cheapest(catalog, slate, domain)
    if role == "last":
        return ctx.last.get(domain)
    if role == "selected":
        return ctx.selected.get(domain)
    return None


def assign_role(eid: str, domain: str, ctx: Context, catalog: Catalog) -> str:
    for role in ("last", "selected", "first", "second", "cheapest"):
        if resolve_role(role, domain, ctx, catalog) == eid:
            return role
    return "other"


def abstract_need(need: Need, ctx: Context, catalog: Catalog) -> AbstractNeed:
    roles = tuple(assign_role(e, need.domain, ctx, catalog) for e in need.entities)
    return AbstractNeed(need.domain, need.op, need.attrs, roles)


def resolve_need(an: AbstractNeed, ctx: Context, catalog: Catalog) -> Optional[Need]:
    """Map a stored need onto the current focal entities; None if it does not resolve."""
    ents = []
    for role in an.roles:
        e = resolve_role(role, an.domain, ctx, catalog)
        if e is None or e in ents:
            return None
        ents.append(e)
    for e in ents:
        if not all(a in catalog.entities[e].attrs for a in an.attrs):
            return None
    return Need(an.domain, an.op, an.attrs, tuple(ents))


def reply_time_continuation(session: Session, i: int, horizon: int, catalog: Catalog) -> list:
    """Next `horizon` turns after reply i, with each need's roles taken in the context after that
    reply, so a role names an entity the service could already see when it planned."""
    ctx = session.contexts[i]
    out = []
    for h in range(1, horizon + 1):
        j = i + h
        if j >= len(session.turns):
            out.append(None)
        elif session.turns[j].need is None:
            out.append(session.label(j))
        else:
            out.append(abstract_need(session.turns[j].need, ctx, catalog))
    return out


def advance(ctx: Context, turn: Turn, m: int) -> Context:
    """Context after the reply at `turn` (policy independent: later turns follow the log)."""
    new = Context(dict(ctx.slates), ctx.focal, dict(ctx.last), dict(ctx.selected))
    for e in turn.named:
        new.selected[turn.domain] = e
        new.last[turn.domain] = e
    if turn.is_search and turn.slate:
        new.slates[turn.domain] = turn.slate
        new.last[turn.domain] = turn.slate[0]
        recent = list(turn.slate) + [e for e in ctx.focal if e not in turn.slate]
        new.focal = tuple(recent)
    elif turn.need is not None:
        if len(turn.need.entities) == 1 and turn.need.entities[0] != UNRESOLVED:
            new.last[turn.domain] = turn.need.entities[0]
    return new


def build_contexts(session: Session, catalog: Catalog, m: int = 3) -> None:
    """Fill session.contexts and each follow-up's abstract need (roles before the turn)."""
    ctx = Context()
    contexts = []
    slate_history: list[tuple[str, ...]] = []
    for turn in session.turns:
        if turn.need is not None:
            turn.abstract = abstract_need(turn.need, ctx, catalog)
        ctx = advance(ctx, turn, m)
        if turn.is_search and turn.slate:
            slate_history.append(turn.slate)
            focal: list[str] = []
            for s in reversed(slate_history[-m:]):
                focal.extend(e for e in s if e not in focal)
            ctx.focal = tuple(focal)
        contexts.append(ctx)
    session.contexts = contexts
