# api_orchestration x a real model: a live run through the engine loop

Primary model: `glm-4-7-flash (via TensorMux)` &middot; fallback: `gpt-5-nano (via OpenAI)` &middot; domain: `api_orchestration` &middot; 12 tasks &middot; **96 real API calls made** (96 primary / 0 fallback)

A weak model was chosen deliberately, not as a fallback: a strong baseline agent leaves the loop no headroom, so every mutation lands in the noise. A weak model gives a low baseline with real room to improve, and it is cheap and fast enough to run several times per task so run-to-run variance means something.

## Noise floor

The root spec was run through the real suite 3 independent times before any mutation, to measure how much `mean_score` -- the metric the loop's own accept/revert decision is made on -- moves on its own, from model sampling alone.

- replicate `mean_score` values: [0.029166666666666664, 0.05833333333333333, 0.17499999999999996]
- population variance: 0.003970
- population std (the noise floor): 0.063007

Any lineage delta below this std is flagged `within_noise_floor` below and must not be read as a real improvement or regression.

## Baseline vs. final (accuracy always paired with cost)

| | accuracy | reliability (variance) | cost (mean tokens/run) | speed (mean seconds/run) |
|---|---|---|---|---|
| baseline (`api-orchestration-gpt5-nano`) | 0.0000 (n=36) | 0.0000 (n=12) | 234.6667 (n=36) | 3.2282s (n=36) |
| final (`api-orchestration-gpt5-nano`) | 0.0000 (n=36) | 0.0000 (n=12) | 234.6667 (n=36) | 3.2282s (n=36) |

Both rows are computed by the real, frozen `agent_engineer.evaluation.Evaluator` harness, `repeats=3`, over the exact same recorded trajectories the noise-floor measurement used -- not hand-computed here.

## The real lineage, verbatim

| gen | dominant cause | mutation | before | after | delta | decision | within noise floor |
|---|---|---|---|---|---|---|---|
| 1 | premature_stop | system_prompt_rewrite (system_prompt) | 0.0875 | 0.1167 | +0.0292 | reverted | yes |
| 2 | premature_stop | strategy_changed (strategy) | 0.0875 | 0.0875 | +0.0000 | reverted | yes |
| 3 | premature_stop | stopping_adjusted (stopping.max_steps) | 0.0875 | 0.1167 | +0.0292 | reverted | yes |
| 4 | premature_stop | memory_reconfigured (memory.kind) | 0.0875 | 0.0875 | +0.0000 | reverted | yes |

Rationale behind each proposed mutation:

- **gen 1** (reverted): The dominant cause is premature_stop. Adding explicit behavioural guidance for it is the smallest edit that can change the agent's conduct at the step where it fails.
- **gen 2** (reverted): Prompt-level guidance did not fix premature_stop, so the loop shape is the constraint: plan_then_execute gives the agent an explicit step for the deliberation it is currently skipping.
- **gen 3** (reverted): Runs are being cut off at 8 steps before finishing, so the budget is binding on the outcome rather than the agent's competence. Doubling it separates the two.
- **gen 4** (reverted): Full retention did not stop the context loss, so the problem is finding the relevant earlier fact, not storing it. Retrieval over 6 relevant items targets that directly.

## Floor Effect Analysis

- **Zero-accuracy baseline**: Baseline accuracy is strictly 0.0000 across all 36 evaluation runs (12 tasks &times; 3 repeats). Multi-step API orchestration tasks inherently require calling external API tools and threading returned identifiers across multiple hops (e.g. customer &rarr; order &rarr; tracking &rarr; scan). Because the single-shot model backend does not execute intermediate tool calls, the model cannot satisfy the required dependency chains, resulting in zero passes.
- **Diagnoser behaviour under zero signal**: With all tasks failing and no intermediate tool calls executed, the diagnose stage consistently attributed failures to `premature_stop` (the agent answered directly with 0 steps, leaving available step budget unused).
- **Mutator escalation**: The mutator appropriately escalated through four distinct rungs targeting `premature_stop`:
  1. System prompt behavioural guidance
  2. Strategy transition (`react` &rarr; `plan_then_execute`)
  3. Step budget doubling (`max_steps: 8` &rarr; `16`)
  4. Memory configuration (`full_transcript` &rarr; `vector_retrieval`)
- **Noise floor protection**: Small positive score deltas (+0.0292 in gen 1 and gen 3 from partial-credit token matching in final answers) were strictly below the measured noise floor (std 0.063007). The selection policy correctly reverted all four mutations, demonstrating that the noise-floor mechanism successfully prevents false-positive adoptions on zero-signal inputs.
