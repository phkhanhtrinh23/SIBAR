# Replay protocol

This page restates the protocol of Section 5.1 and Appendix B of the paper and points to the code
that implements each part. The last section lists where the implementation fills in details the
paper does not give, or departs from the text.

## Datasets and splits

* **CrossWOZ** (`sibar/data/crosswoz.py`). The paper uses 3 of CrossWOZ's 5 domains for recommendation: Beijing attractions, restaurants and hotels.
  * The 61 official test seekers all appear in training, so the data are split by seeker.
  * The split has 64 training and 18 test seekers, which gives 575 test sessions with no seeker overlap.
  * The development split is 20% of the CrossWOZ training seekers.
* **MultiWOZ 2.2** (`sibar/data/multiwoz.py`). Uses the official split. Each dialogue is one bootstrap unit.
* **VOGUE** (`sibar/data/vogue.py`). 60 dialogues from 10 seekers, over 3 catalogs with 36 items in total.
  * Two annotators independently ground each follow-up to an operator, an entity scope and attributes.
  * VOGUE reuses the CrossWOZ configuration.

## Turn labels and needs (Appendix B)

* **Search turn.** A user turn with inform or select acts is a search turn, unless its only inform
  act names a shown entity and asks for its attributes.
* **Stricter labeling.** The stricter labeling in Table 4 counts every turn with an inform act as a search.
* **Need attributes.** A need's attributes are the requested (domain, slot) pairs on the focal entities.
* **Stored needs.** Needs are stored as (operator, entity role), such as (price, cheapest), so
  scenarios carry no other owner's text (`sibar/roles.py`).
* **coverage@4** (`sibar/analysis.py`). Measures whether a label-frequency model, from the reply-time state, has the next need among its top four labels.

## Session state and facts

* `E^ses_t` holds the items of the current answer and the last `m = 3` replies (`sibar/pool.py`).
* Relational summaries keep the top 3 neighbors by rating and distance, within 32 tokens.
* The pool is capped at `M = 32` facts by schema priority.
* Churn changes only volatile attributes (price, rating, hours, availability), at a per-turn rate
  (`sibar/catalog.py::ChurnProcess`).

## Receiver, costs and uncertainty

* **Agent.** One frozen agent, Qwen2.5-14B-Instruct at temperature 0 (`sibar/agent.py`).
  * It sees the owner's question, the last answer and the cached facts in `R̂(q)`.
  * It cites a fact for every value it states.
  * Each cited value is checked against the catalog after normalizing numbers and units.
* **Correctness.** A local answer is correct if it states every requested value and each matches
  the catalog. A follow-up the agent cannot answer is a call, and later turns follow the log
  (`sibar/simulator.py`).
* **Metrics** (`sibar/metrics.py`).
  * **Avoid**: the share of follow-ups answered locally and correctly.
  * **Unsup**: the share answered locally with a wrong or unsupported answer.
  * **Tok**: packet tokens per session.
  * **Net**: `ΔJ` (Eq. 2) as a percentage of the cost of calling on every follow-up, charging `C_err` for every wrong local answer.
* **Costs.**
  * `C_call = 1` (420 ms) and `C_err = 2`.
  * `λ = 0.0012` per token, which prices about 830 tokens at one call.
  * `C_pred = 0.038` per non-empty packet, for retrieval and planning.
  * The budget is `B = 128` tokens per reply.
* **Lease cap.** A 30-minute lease cap, the usual Web-session timeout, never binds, because replays lack timestamps.
* **Point estimates.** Each averages 40 replays per session over seeds.
* **Intervals.** 1,000 paired bootstrap resamples over the 18 CrossWOZ seekers, the 10 VOGUE seekers or the MultiWOZ 2.2 dialogues.
  * Tuned configurations are held fixed.
  * Offsets are scaled by `t_{G−1,0.975}/z_{0.975}` for `G` seekers (1.08 for CrossWOZ, 1.15 for VOGUE).

## Baselines (`sibar/policies.py`)

All prior methods use the same catalog and serving gate. They build their candidates without Sibar's
cache-aware pool (C1), so a packet can repeat facts the agent already holds.

| Method | Packet |
|--------|--------|
| `Compact-Record` | A fixed attribute record of the focal entities, up to `B`. |
| `CFQP-Pkg` | Its top two predicted questions, from the same `S` neighbors as Sibar. |
| `Spec-Action` | The result of its top predicted lookup. |
| `Trace-Bundle` | The most frequent request chains of length up to `H`. |
| `ProAct-Pkg` | Up to three needs predicted with confidence at least 0.6. |

**Send gate.** Each sends a packet only if the sum, over the needs it serves, of the baseline's own
probability for each need times `v_A` exceeds `λ c(P) + C_pred` (`PacketBaseline.gate`).

## Tuning and ablations (`scripts/run.py`)

* **Tuning target.** Each policy is tuned for Net with at most 16 configurations on the development split.
* **Grids.** Sibar tries `H ∈ {2, 3, 4, 6}` and `γ ∈ {0.5, 0.7, 0.9, 1.0}`; Reply-Aware tries the same `H`.
* **Re-tuning.** Both are re-tuned for each budget and churn rate, and `γ` for each `H`.
* **Test-set use.** The sweeps and the cost sensitivity table use test sessions after the defaults were fixed.
* **Latency.** LP bounds use HiGHS. Latency is timed per reply, on one CPU thread, from the embedded state to the packet.
* **Ablation variants.** The greedy ablation adds requirement sets by value per token. The no-digest ablation plans as if the agent's cache were empty.

## Implementation notes

These points are where the code fills in details the paper does not give, or departs from it.

* **Question parsing** uses the gold need, `R̂(q) = R(n)`, as the paper takes parsing as given.
* **Seeker split.** The CrossWOZ seeker split uses a salted SHA-256 hash of the seeker id. The
  usability head is fit on a further 10% of the training seekers.
* **Unresolved follow-ups.** A follow-up whose need does not resolve to catalog entities keeps its
  follow-up label with the entity `<unresolved>` (`sibar/types.py::UNRESOLVED`). It is always a call.
* **CrossWOZ-specific rules.**
  * A request for the nearby metro station maps to the `地铁` attribute of the named entity.
  * Taxi turns are templated and are always search turns.
  * Hotel amenities become boolean facts `酒店设施-X`.
* **Relational summaries** are ranked by rating only, because the CrossWOZ database has no distances.
* **Volatile attributes** are prices, ratings and opening hours on CrossWOZ, and price,
  availability and rating on VOGUE.
* **Token cost.** `c(f)` counts tokens with `SIBAR_TOKENIZER`; the paper protocol uses the agent's
  tokenizer.
* **Last answer.** It is the logged system reply before the follow-up, cut to 600 characters. When
  an earlier follow-up was answered locally, the next prompt still shows the logged reply.
* **Declined answers.** An agent that passes the serving gate but cites nothing is counted as a wrong
  local answer, so it adds to Unsup.
* **Spec-Action** takes its top predicted lookup from the backoff chain and sends that lookup's
  attributes for every entity of the slate.
* **Trace-Bundle** sends the single most frequent chain, and gives each need in it that chain's
  frequency as its probability. The paper describes several chains and a probability for each
  need.
* **ProAct-Pkg.** With `PREDICTOR_BACKEND=oracle`, it uses a label-frequency stand-in for the LLM,
  flagged in the logs.
* **VOGUE.** The dialogues and their grounding are not redistributed. The loader documents the
  expected JSON format.
