# extraction x a real model: a live run through the engine loop

Primary model: `glm-4-7-flash (via TensorMux)` &middot; fallback: `gpt-5-nano (via OpenAI)` &middot; domain: `extraction` &middot; 14 tasks &middot; **154 real API calls made** (154 primary / 0 fallback)

A weak model was chosen deliberately, not as a fallback: a strong baseline agent leaves the loop no headroom, so every mutation lands in the noise. A weak model gives a low baseline with real room to improve, and it is cheap and fast enough to run several times per task so run-to-run variance means something.

## Noise floor

The root spec was run through the real suite 3 independent times before any mutation, to measure how much `mean_score` -- the metric the loop's own accept/revert decision is made on -- moves on its own, from model sampling alone.

- replicate `mean_score` values: [0.9136054285714286, 0.9517007142857142, 0.9255102142857143]
- population variance: 0.000253
- population std (the noise floor): 0.015913

Any lineage delta below this std is flagged `within_noise_floor` below and must not be read as a real improvement or regression.

## Baseline vs. final (accuracy always paired with cost)

| | accuracy | reliability (variance) | cost (mean tokens/run) | speed (mean seconds/run) |
|---|---|---|---|---|
| baseline (`extraction-gpt5-nano`) | 0.8571 (n=42) | 0.0317 (n=14) | 612.9762 (n=42) | 3.8486s (n=42) |
| final (`extraction-gpt5-nano-g4`) | 0.8095 (n=42) | 0.0476 (n=14) | 613.6905 (n=42) | 3.6980s (n=42) |

Both rows are computed by the real, frozen `agent_engineer.evaluation.Evaluator` harness, `repeats=3`, over the exact same recorded trajectories the noise-floor measurement used -- not hand-computed here.

## The real lineage, verbatim

| gen | dominant cause | mutation | before | after | delta | decision | within noise floor |
|---|---|---|---|---|---|---|---|
| 1 | output_format_violation | system_prompt_rewrite (system_prompt) | 0.9136 | 0.9279 | +0.0143 | reverted | yes |
| 2 | output_format_violation | strategy_changed (strategy) | 0.9136 | 0.9279 | +0.0143 | reverted | yes |
| 3 | output_format_violation | stopping_adjusted (stopping.max_steps) | 0.9136 | 0.8446 | -0.0690 | reverted | no |
| 4 | output_format_violation | memory_reconfigured (memory.kind) | 0.9136 | 0.9398 | +0.0262 | accepted | no |

Rationale behind each proposed mutation:

- **gen 1** (reverted): The dominant cause is output_format_violation. Adding explicit behavioural guidance for it is the smallest edit that can change the agent's conduct at the step where it fails.
- **gen 2** (reverted): Prompt-level guidance did not fix output_format_violation, so the loop shape is the constraint: plan_then_execute gives the agent an explicit step for the deliberation it is currently skipping.
- **gen 3** (reverted): Runs are being cut off at 8 steps before finishing, so the budget is binding on the outcome rather than the agent's competence. Doubling it separates the two.
- **gen 4** (accepted): Full retention did not stop the context loss, so the problem is finding the relevant earlier fact, not storing it. Retrieval over 6 relevant items targets that directly.
