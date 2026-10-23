# Method-to-code map

This page maps each part of Sections 3–4 and Appendix A–B of the paper to the code that
implements it. Default values come from `configs/default.yaml`.

## Data model (Sec. 3)

| Paper | Code |
|-------|------|
| Fact `f = (e, a, v, s_f)` with a provenance signature | `sibar/types.py`, `sibar/catalog.py` (HMAC signature, `SIGNATURE_TOKENS = 4` in `sibar/tokens.py`) |
| Need `n = (κ, Ω)`, requirement `R(n)` | `Need` in `sibar/types.py` |
| Token cost `c(f)` | `fact_cost` in `sibar/tokens.py`: tokens of `name\|attr\|value` plus the signature |
| Lease `Δ(a)`, joint freshness `φ` | `sibar/freshness.py` |
| Local value `v_A` (Def. 2) | `local_value` in `sibar/freshness.py` |
| Realized objective `J` (Eq. 2) | `SessionResult.J` in `sibar/simulator.py` |
| Churn of volatile attributes | `ChurnProcess` in `sibar/catalog.py` |

## C1 — Cache-aware pool (Eq. 4)

`sibar/pool.py` builds the pool `F_t`:
* It covers the facts of the focal entities: the current answer plus the last `m = 3` replies.
* It skips static facts the agent already holds, using the cache digest `δ_t` from `sibar/cache.py`.
* Volatile facts stay candidates for a refresh.
* It caps the pool at `M = 32` facts by schema priority, i.e. how often training owners asked about that (domain, attribute).

Relational summaries keep the top 3 neighbors by rating and distance, within 32 tokens.

## C2 — Cross-user trajectory sampler

`sibar/sampler.py`, `sibar/encoder.py`, `sibar/roles.py`:

* Reply states are embedded with `paraphrase-multilingual-MiniLM-L12-v2` and indexed with HNSW
  (16 links, search breadth 64).
* `S = 16` nearest training states are weighted by `exp(⟨ψ, ψ'⟩ / τ_w)` with `τ_w = 0.1`. Their
  continuations become scenarios.
* Needs are stored as (operator, entity role), so a scenario carries no other owner's text.
  `resolve_need` maps a role onto the current focal entities.
* The backoff chain (`MarkovChain`) has three levels:
  * the turn label;
  * its (kind, domain) pair;
  * label frequencies.

  Each level is kept with weight `n/(n+2)`. The neighbor mixture uses `α = N(z)/(N(z)+η)` with `η = 10`.

## C3 — Receding-horizon planner (Sec. 4.3, Appendix A)

`sibar/planner.py`:

| Credit | Paper | `credit` / config |
|--------|-------|-------------------|
| One-step | Eq. 3 | `horizon = 1` |
| Horizon | `V_H` (Eq. 5), discount `γ` | `credit: horizon` |
| Reply-aware | `V_H^ra` (Prop. horizon (iii)) | `credit: reply_aware` |

* **Exact selector** (`selector: exact`): branch and bound over the include/exclude tree, with
  exact leaf scores and the LP relaxation of Eq. 6 solved by HiGHS.
* **Greedy selector** (`selector: greedy`): used in the ablation.
* **Optimality test:** `tests/test_propositions.py` checks Proposition 3 (B&B optimality) against brute force.

## C4 — Verified agent cache (Sec. 4.4)

`sibar/agent.py`:

* **Serving gate.** The agent answers locally only if every fact in `R(n)` has a lease-valid copy
  (`complete_evidence_gate`) and `v_A ≥ 0`. Otherwise it calls the service.
* **LLM answerer.** It sees the owner's question, the last answer and the cached facts in `R̂(q)`.
  It must cite a fact for every value it states. Each cited value is checked against the catalog
  after normalizing numbers and units.
* **Usability head.** `u_A(n) = logistic(β_0(κ_n) − β_1 |R(n)|)`. It is fit on a separate fold of training
  sessions with the frozen agent, where `u_A(n)` is the probability that the agent answers `n`
  correctly from complete, current evidence.

## Replay (Algorithm 1, lines 6–13)

`sibar/simulator.py::replay_session` replays one logged session turn by turn:

1. **At a search turn**, the reply states the logged answer.
2. **At a follow-up**, the agent tries a local answer:
   * If it answers, the result is scored correct or wrong and no call is made.
   * Otherwise the follow-up becomes a call, and the fallback answer states `R(n)`.
3. **Every reply** (search answer or call) carries a packet planned by the policy. The packet is
   cached with its leases.
4. **Later turns** follow the log.

## Policies

`sibar/policies.py`, `sibar/experiment.py::method_config`:

| Method | Construction |
|--------|-------------|
| `Sibar` | C1 + C2 + C3 with horizon credit; `H`, `γ` tuned on dev |
| `One-Step` | same, `H = 1` |
| `Reply-Aware` | same, reply-aware credit, `γ = 1` |
| `Hindsight *` | the true continuation is the only scenario |
| `Reply-Only` | no packet |
| Prior methods | see `docs/protocol.md` |
