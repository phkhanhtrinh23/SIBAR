# Beyond the First Question: Multi-Step Information Bundling in Recommender Systems for User-Delegated Agents

This is the official code for the paper "Beyond the First Question: Multi-Step Information
Bundling in Recommender Systems for User-Delegated Agents".
The paper introduces **Sibar** (**S**peculative **I**nformation **B**undling for
**A**gent-mediated **R**ecommendation). When a recommendation service replies to a user's agent,
Sibar attaches a budgeted **packet** of facts about the recommended items. The agent can then
answer the owner's next follow-up questions from its cache instead of calling the service again.

Sibar works in four parts:
* It predicts what the owner will ask next from other owners' sessions.
* It credits a packet with every predicted need it serves within H turns.
* It selects the packet exactly, by branch and bound, within the per-reply token budget.
* It replans at every reply.

<p align="center">
  <img src="figures/fig_teaser.png" alt="Follow-ups about two recommended hotels. A traditional recommender returns items only, so each question costs a call. Packets for the next question (One-Step) or up to the next reply (Reply-Aware) still miss transit. Sibar plans facts several questions ahead, so no follow-up needs a call." width="720" />
</p>
<p align="center"><em>Follow-ups about two recommended hotels. A traditional recommender returns items only, so each question costs a call. Packets for the next question (One-Step) or up to the next reply (Reply-Aware) still miss transit. Sibar plans facts several questions ahead, so no follow-up needs a call.</em></p>

---

## 1. Repo overview

```
code/
  configs/default.yaml        # Paper defaults (Sec. 5.1, Appendix B)
  scripts/run.py              # Single entry point: one sub-command per table / figure
  scripts/run_all.sh          # Every experiment of Section 5 in sequence (tune -> test)
  sibar/
    types.py, catalog.py      # Data model: fact (e, a, v, s_f), need, turn, session; churn
    freshness.py              # Leases Δ(a), joint freshness φ, local value v_A (Def. 2)
    cache.py                  # Agent cache M, verified copies M✓_t, cache digest δ_t
    pool.py                   # C1  cache-aware fact pool F_t (Eq. 4)
    sampler.py, encoder.py,
    roles.py                  # C2  cross-user trajectory sampler (HNSW + backoff chain)
    planner.py                # C3  V_H / V_H^ra / one-step credit, branch and bound + LP bound (Eq. 6)
    agent.py                  # C4  verified cache, complete-evidence gate, usability head, answer check
    policies.py               # Sibar, internal controls, Reply-Only and the five adapted prior methods
    simulator.py              # Whole-session replay (Algorithm 1, lines 6-13)
    metrics.py                # Avoid, Unsup, Tok, Net; paired seeker bootstrap
    analysis.py               # Follow-up run statistics and coverage@4 (Table 4)
    data/                     # CrossWOZ, MultiWOZ 2.2, VOGUE and synthetic loaders
    llm/                      # OpenAI-compatible backend (OpenRouter or local vLLM) with disk cache
  tests/                      # Example 1, Proposition 1, Proposition 3 (B&B optimality)
  figures/                    # Figures from the paper used in this README
  docs/                       # Method-to-code map and replay protocol details
```

## 2. Method recap

Sibar has four components. `docs/method.md` maps each one to its file and equation.

<p align="center">
  <img src="figures/fig_overview.png" alt="Sibar between a recommendation service and a user's agent. At each reply the service gathers facts the agent lacks or must refresh (C1), predicts the next turns from similar past sessions (C2) and sends the packet with the highest expected net saving within the budget (C3). The agent answers from valid facts or calls back (C4)." width="100%" />
</p>
<p align="center"><em>Sibar between a recommendation service and a user's agent. At each reply the service gathers facts the agent lacks or must refresh (C1), predicts the next turns from similar past sessions (C2) and sends the packet with the highest expected net saving within the budget (C3). The agent answers from valid facts or calls back (C4).</em></p>

| Symbol | Component | What it does | Code |
|--------|-----------|--------------|------|
| `C1` | Cache-aware pool | Gathers facts about the recommended items, skips static facts the agent already holds, attaches leases from catalog update rates, and caps the pool at `M = 32`. | `pool.py` |
| `C2` | Cross-user trajectory sampler | Retrieves the `S = 16` nearest past reply states (HNSW over multilingual MiniLM embeddings) and turns their continuations into scenarios of the next turns. The backoff chain `α = N/(N+η)` covers sparse states. | `sampler.py` |
| `C3` | Receding-horizon planner | Credits a packet with the predicted needs it serves within `H` turns, discounted by `γ`, and selects it exactly within budget `B` by branch and bound with an LP bound. | `planner.py` |
| `C4` | Verified agent cache | Answers locally only from complete, lease-valid evidence when `v_A ≥ 0`. Otherwise it calls the service. | `agent.py` |

The methods compared in Section 5 (`sibar/experiment.py`):

| Group | Methods |
|-------|---------|
| Floor | `Reply-Only` |
| Prior methods, adapted to fact packets | `Compact-Record`, `CFQP-Pkg`, `Spec-Action`, `Trace-Bundle`, `ProAct-Pkg` |
| Internal controls (same pool, scenarios and selector; differ only in credit) | `One-Step`, `Reply-Aware`, `Sibar` |
| Hindsight (true continuation as the only scenario) | `Hindsight One-Step`, `Hindsight Sibar` |

### Replay loop (ASCII)

```
                ┌──────────────────────────────────────────────────────────┐
  turn t of  ──►│ search turn : the service replies with the answer        │
  a logged      │ follow-up   : agent answers from cache if Suff·I[v_A≥0]  │
  session       │               holds, else it calls (a call is a reply)   │
                │ every reply : C1 pool → C2 scenarios → C3 packet P_t     │
                │               (c(P_t) ≤ B), cached with leases            │
                └──────────────────────────────────────────────────────────┘
                          │  later turns follow the log
                          ▼
             per session: correct / wrong local answers, calls, packet tokens
             → Avoid, Unsup, Tok, Net (ΔJ as % of calling on every follow-up)
```

The loop is implemented in `sibar/simulator.py`.

## 3. Environment

| Component | Requirement | Tested with |
|-----------|-------------|-------------|
| Python | ≥ 3.10 | 3.11 |
| NumPy | ≥ 2.0 | 2.4.6 |
| SciPy (HiGHS LP) | ≥ 1.11 | 1.17.1 |
| scikit-learn | ≥ 1.3 | 1.9.1 |
| hnswlib | ≥ 0.8 | 0.8.0 |
| sentence-transformers | ≥ 3.0 | 6.2.0 |
| PyTorch | any recent | 2.14.1 |
| tiktoken / transformers | ≥ 0.7 / any | 0.14.0 / 5.20.2 |
| openai SDK | ≥ 1.0 | 3.37.0 |
| vLLM *(optional, local agent)* | — | 0.22.1 |

* The GPU is only used by the state encoder and the optional local LLM server. The planner runs on CPU.
* The first run downloads the encoder `paraphrase-multilingual-MiniLM-L12-v2` from HuggingFace.
* Without network access, pass `--encoder hash`.

## 4. Install

```bash
conda create -n sibar python=3.11 -y && conda activate sibar
pip install -r requirements.txt
cp .env.example .env
```

Environment variables (`.env`, git-ignored):

| Variable | Purpose | Default |
|----------|---------|---------|
| `AGENT_BACKEND` | Frozen agent (C4): `oracle` (deterministic, no API) or `openrouter` (any OpenAI-compatible endpoint). | `oracle` |
| `AGENT_MODEL`, `AGENT_BASE_URL` | Agent model id and endpoint. | `qwen/qwen-2.5-72b-instruct`, OpenRouter |
| `PREDICTOR_BACKEND`, `PREDICTOR_MODEL`, `PREDICTOR_BASE_URL` | LLM for the `ProAct-Pkg` baseline (`oracle` = label-frequency stand-in). | `oracle` |
| `OPENROUTER_API_KEY` | Key for the endpoint. It must be non-empty for `openrouter`; a local server ignores its value. | *(unset)* |
| `SIBAR_TOKENIZER` | Tokenizer for fact costs `c(f)`: a tiktoken encoding or a HuggingFace id. | `cl100k_base` |

**Paper protocol.** The agent is Qwen2.5-14B-Instruct at temperature 0, and tokens are counted
with its own tokenizer. OpenRouter does not serve the 14B model, so serve it locally:

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct --port 8765 --max-model-len 8192 --seed 0
```

Then uncomment the "Paper protocol" block of `.env.example` in your `.env`.

Every completion is cached in `.cache/llm_cache.sqlite`, so repeated replays and reruns cost no
extra calls.

## 5. Datasets

| Name | Source | Split | Place raw files in |
|------|--------|-------|--------------------|
| CrossWOZ | [thu-coai/CrossWOZ](https://github.com/thu-coai/CrossWOZ) | train/val/test merged, split by seeker; 20% of training seekers as dev | `data/raw/crosswoz_repo/` |
| MultiWOZ 2.2 | [budzianowski/multiwoz](https://github.com/budzianowski/multiwoz) | official split; each dialogue is a bootstrap unit | `data/raw/multiwoz_repo/` |
| VOGUE | [D3Mlab/vogue](https://github.com/D3Mlab/vogue) | leave-one-seeker-out | `data/raw/vogue/vogue.json` |

```bash
mkdir -p data/raw && cd data/raw
git clone --depth 1 https://github.com/thu-coai/CrossWOZ.git crosswoz_repo
git clone --depth 1 https://github.com/budzianowski/multiwoz.git multiwoz_repo
cd ../..
python scripts/run.py prepare --dataset crosswoz     # -> data/processed/crosswoz.pkl
python scripts/run.py prepare --dataset multiwoz
```

* **CrossWOZ.** Seekers are assigned to train or test by a salted SHA-256 hash of the seeker id,
  so the split is stable and needs no split files.
* **VOGUE.** Download the dialogues from the link above. Their hand grounding is not part of this repo; convert them to the
  JSON format documented in `sibar/data/vogue.py`.
* **Turn labels.** Turns are labeled from dialogue acts as described in Appendix B.
* `docs/protocol.md` gives the exact rules: turn labels, follow-ups whose need is unresolved,
  logged replies, facts and their token costs.

## 6. How to run

`scripts/run.py` is the single entry point. Each sub-command reproduces one table or figure and
writes to `outputs/<dataset>/`. Results are not tracked in git.

| Command | Paper | Output |
|---------|-------|--------|
| `runs` | Table 4 (follow-up runs, coverage@4) | `table4_runs.json` |
| `tune [--param budget\|horizon\|churn]` | `H`, `γ` chosen on the dev split (Appendix B) | `tune*.json` |
| `main` | Table 5, Fig. 3 | `main.jsonl`, `table5_main.json` |
| `reprice` | Table 6 (re-prices the packets of `main`) | `table6_reprice.json` |
| `ablation` | Fig. 4b | `fig4b_ablation.json` |
| `sweep --param horizon\|budget\|oracle_mix\|gamma` | Fig. 5 | `fig5_<param>.json` |
| `churn` | Fig. 6 | `fig6_churn.json` |
| `latency` | planning latency per reply | console |

Run `tune` before `main`, because `main` reads the tuned `H` and `γ` from `tune.json`:

```bash
python -m pytest -q tests                                   # Example 1, Prop. 1, Prop. 3
python scripts/run.py runs --dataset crosswoz
python scripts/run.py tune --dataset crosswoz --replays 5 --workers 16
python scripts/run.py main --dataset crosswoz --replays 40 --workers 16
python scripts/run.py reprice --dataset crosswoz
```

| Flag | Meaning |
|------|---------|
| `--dataset` | `crosswoz`, `multiwoz`, `vogue` or `synthetic`. |
| `--replays` | Replays per session over seeds (paper: 40; default 1). |
| `--workers` | Worker processes. |
| `--max-sessions` | Subsample test sessions for a quick run (0 = all). |
| `--methods` | Comma-separated subset of methods for `main`. |
| `--config` | YAML config (default `configs/default.yaml`). |
| `--encoder` | Override the state encoder (`hash` runs offline). |

### Full pipeline

```bash
bash scripts/run_all.sh 32 40                 # [workers] [replays]; logs in outputs/logs/
FROM=crosswoz_main bash scripts/run_all.sh    # resume from a step
PY=/path/to/python bash scripts/run_all.sh    # pick the interpreter
```

`main` prints one row per method plus the paired contrasts against `Sibar`, for example:

```
Method                   Avoid   Unsup     Tok     Net   Calls   Net 95% CI
Reply-Only                 ...
...
Sibar                      ...
Sibar - One-Step: Avoid +.. [.., ..]  Net +.. [.., ..]
Sibar - Reply-Aware: Avoid +.. [.., ..]  Net +.. [.., ..]
```
