"""Builds the components for a dataset and replays sessions under each method (Sec. 5)."""
from __future__ import annotations

import dataclasses
import json
import multiprocessing as mp
import os
from typing import Callable, Optional

from tqdm import tqdm

from .agent import Agent, LLMAnswerer, OracleAnswerer, fit_usability
from .config import SibarConfig
from .data.common import Dataset
from .encoder import StateEncoder
from .llm import get_backend
from .policies import (CFQPPkg, CompactRecord, Policy, ProActPkg, ReplyOnly, Sibar, SpecAction,
                       TraceBundle)
from .sampler import TrajectorySampler, state_text
from .simulator import SessionResult, replay_session
from .tokens import DEFAULT_TOKENIZER, tokenizer_name

PRIOR = ["Compact-Record", "CFQP-Pkg", "Spec-Action", "Trace-Bundle", "ProAct-Pkg"]
INTERNAL = ["One-Step", "Reply-Aware", "Sibar"]
HINDSIGHT = ["Hindsight One-Step", "Hindsight Sibar"]
ALL_METHODS = ["Reply-Only"] + PRIOR + INTERNAL + HINDSIGHT


def method_config(name: str, cfg: SibarConfig) -> SibarConfig:
    """One-Step and Reply-Aware share Sibar's pool, scenarios and exact selector; only credit differs."""
    if name in ("One-Step", "Hindsight One-Step"):
        cfg = cfg.replace(horizon=1, credit="horizon")
    elif name == "Reply-Aware":
        cfg = cfg.replace(credit="reply_aware", gamma=1.0)
    elif name in ("Sibar", "Hindsight Sibar"):
        cfg = cfg.replace(credit="horizon")
    if name.startswith("Hindsight"):
        cfg = cfg.replace(scenario_source="hindsight")
    return cfg


def load_dataset(name: str, raw_dir: str = "data/raw", processed_dir: str = "data/processed",
                 rebuild: bool = False, **kw) -> Dataset:
    tok = tokenizer_name()                       # relational summaries are cut to a token budget
    suffix = "" if tok == DEFAULT_TOKENIZER else "." + tok.split("/")[-1]
    path = os.path.join(processed_dir, f"{name}{suffix}.pkl")
    if os.path.exists(path) and not rebuild:
        return Dataset.load(path)
    if name == "crosswoz":
        from .data import crosswoz
        ds = crosswoz.load(raw_dir, **kw)
    elif name == "multiwoz":
        from .data import multiwoz
        ds = multiwoz.load(raw_dir, **kw)
    elif name == "synthetic":
        from .data import synthetic
        ds = synthetic.load(**kw)
    elif name == "vogue":
        from .data import vogue
        ds = vogue.load(raw_dir, **kw)
    else:
        raise ValueError(name)
    ds.save(path)
    return ds


class Runner:
    def __init__(self, ds: Dataset, cfg: SibarConfig, cache_dir: str = ".cache",
                 encoder: Optional[str] = None, log: Callable = print):
        self.ds = ds
        self.cfg = cfg
        self.log = log
        self.encoder = StateEncoder(encoder or cfg.encoder, cache_dir)
        backend = get_backend("AGENT", cache_dir)
        self.answerer = LLMAnswerer(backend) if backend is not None else OracleAnswerer()
        self.predictor = get_backend("PREDICTOR", cache_dir)
        if self.predictor is None:
            log("[info] PREDICTOR_BACKEND=oracle: ProAct-Pkg uses a label-frequency stand-in for the LLM")
        log(f"[info] agent: {'LLM ' + backend.model if backend else 'oracle (deterministic)'}")
        self.usability = fit_usability(ds.usability_fold, ds.catalog, self.answerer)
        log(f"[info] usability head: {self.usability.to_dict()}")
        texts = [state_text(s, i) for s in ds.train + ds.dev + ds.test for i in range(len(s.turns))]
        self.encoder.encode(texts)                   # pre-encode every reply state once
        self._samplers: dict = {}

    def sampler(self, seed: int, cfg: SibarConfig) -> TrajectorySampler:
        key = (seed, cfg.neighbors, cfg.hnsw_m, cfg.hnsw_ef, cfg.tau_w, cfg.eta)
        if key not in self._samplers:
            self._samplers[key] = TrajectorySampler(cfg, self.encoder).fit(self.ds.train, seed=seed,
                                                                     catalog=self.ds.catalog)
        s = self._samplers[key]
        s.cfg = cfg
        return s

    def policy(self, name: str, seed: int, cfg: Optional[SibarConfig] = None) -> tuple[Policy, SibarConfig]:
        cfg = method_config(name, cfg or self.cfg)
        cat, u = self.ds.catalog, self.usability
        smp = self.sampler(seed, cfg)
        if name == "Reply-Only":
            return ReplyOnly(), cfg
        if name in INTERNAL + HINDSIGHT or name.startswith("Sibar") or name.startswith("Ablation"):
            return Sibar(name, cfg, cat, smp, u), cfg
        if name == "Compact-Record":
            return CompactRecord(name, cfg, cat, u, chain=smp.chain), cfg
        if name == "CFQP-Pkg":
            return CFQPPkg(name, cfg, cat, u, sampler=smp), cfg
        if name == "Spec-Action":
            return SpecAction(name, cfg, cat, u, chain=smp.chain), cfg
        if name == "Trace-Bundle":
            return TraceBundle(name, cfg, cat, u, train=self.ds.train, horizon=cfg.horizon), cfg
        if name == "ProAct-Pkg":
            return ProActPkg(name, cfg, cat, u, predictor=self.predictor, chain=smp.chain), cfg
        raise ValueError(name)

    # --- running -------------------------------------------------------------
    def run(self, methods: list, sessions: Optional[list] = None, replays: int = 1, workers: int = 1,
            cfg_for: Optional[dict] = None, desc: str = "replay") -> list[SessionResult]:
        """`methods` are names, or (label, base name, cfg) triples for variants."""
        sessions = self.ds.test if sessions is None else sessions
        jobs = []
        for m in methods:
            label, base, cfg = (m, m, (cfg_for or {}).get(m, self.cfg)) if isinstance(m, str) else m
            for r in range(replays):
                jobs.append((label, base, cfg, r))
        global _RUNNER, _SESSIONS
        _RUNNER, _SESSIONS = self, sessions
        for _, base, cfg, r in jobs:                 # build samplers before forking
            self.sampler(r, method_config(base, cfg))
        tasks = [(j, k) for j in range(len(jobs)) for k in range(0, len(sessions), 50)]
        results: list[SessionResult] = []
        if workers > 1:
            ctx = mp.get_context("fork")
            with ctx.Pool(workers) as pool:
                for out in tqdm(pool.imap_unordered(_work, [(jobs[j], k) for j, k in tasks]),
                                total=len(tasks), desc=desc):
                    results.extend(out)
        else:
            for j, k in tqdm(tasks, desc=desc):
                results.extend(_work((jobs[j], k)))
        return results

    def latencies(self, cfg: SibarConfig, sessions: list, seed: int = 0) -> list[float]:
        pol, mcfg = self.policy("Sibar", seed, cfg)
        agent = Agent(self.ds.catalog, mcfg, self.usability, self.answerer)
        for s in sessions:
            replay_session(s, pol, agent, self.ds.catalog, mcfg, seed)
        return pol.latencies


_RUNNER: Optional[Runner] = None
_SESSIONS: list = []


def _work(arg) -> list[SessionResult]:
    (label, base, cfg, r), start = arg
    runner = _RUNNER
    pol, mcfg = runner.policy(base, r, cfg)
    pol.name = label
    agent = Agent(runner.ds.catalog, mcfg, runner.usability, runner.answerer)
    return [replay_session(s, pol, agent, runner.ds.catalog, mcfg, r) for s in _SESSIONS[start:start + 50]]


def save_results(results: list[SessionResult], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(dataclasses.asdict(r), ensure_ascii=False) + "\n")


def load_results(path: str) -> list[SessionResult]:
    with open(path) as f:
        return [SessionResult(**{**json.loads(l), "positions": [tuple(x) for x in json.loads(l)["positions"]]})
                for l in f]
