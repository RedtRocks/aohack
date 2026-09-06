# code_math x a real model: a live run through the engine loop

Primary model: `glm-4-7-flash (via TensorMux)` &middot; fallback: `gpt-5-nano (via OpenAI)` &middot; domain: `code_math` &middot; 16 tasks &middot; **128 real API calls made** (128 primary / 0 fallback)

A weak model was chosen deliberately, not as a fallback: a strong baseline agent leaves the loop no headroom, so every mutation lands in the noise. A weak model gives a low baseline with real room to improve, and it is cheap and fast enough to run several times per task so run-to-run variance means something.

## Noise floor

The root spec was run through the real suite 3 independent times before any mutation, to measure how much `mean_score` -- the metric the loop's own accept/revert decision is made on -- moves on its own, from model sampling alone.

- replicate `mean_score` values: [0.625, 0.6875, 0.675]
- population variance: 0.000729
- population std (the noise floor): 0.027003

Any lineage delta below this std is flagged `within_noise_floor` below and must not be read as a real improvement or regression.

## Baseline vs. final (accuracy always paired with cost)

| | accuracy | reliability (variance) | cost (mean tokens/run) | speed (mean seconds/run) |
|---|---|---|---|---|
| baseline (`code-math-gpt5-nano`) | 0.6458 (n=48) | 0.0417 (n=16) | 352.8125 (n=48) | 3.9191s (n=48) |
| final (`code-math-gpt5-nano`) | 0.6458 (n=48) | 0.0417 (n=16) | 352.8125 (n=48) | 3.9191s (n=48) |

Both rows are computed by the real, frozen `agent_engineer.evaluation.Evaluator` harness, `repeats=3`, over the exact same recorded trajectories the noise-floor measurement used -- not hand-computed here.

## The real lineage, verbatim

| gen | dominant cause | mutation | before | after | delta | decision | within noise floor |
|---|---|---|---|---|---|---|---|
| 1 | premature_stop | system_prompt_rewrite (system_prompt) | 0.7500 | 0.6000 | -0.1500 | reverted | no |
| 2 | premature_stop | strategy_changed (strategy) | 0.7500 | 0.6250 | -0.1250 | reverted | no |
| 3 | premature_stop | stopping_adjusted (stopping.max_steps) | 0.7500 | 0.7500 | +0.0000 | reverted | yes |
| 4 | premature_stop | memory_reconfigured (memory.kind) | 0.7500 | 0.6875 | -0.0625 | reverted | no |

Rationale behind each proposed mutation:

- **gen 1** (reverted): The dominant cause is premature_stop. Adding explicit behavioural guidance for it is the smallest edit that can change the agent's conduct at the step where it fails.
- **gen 2** (reverted): Prompt-level guidance did not fix premature_stop, so the loop shape is the constraint: plan_then_execute gives the agent an explicit step for the deliberation it is currently skipping.
- **gen 3** (reverted): Runs are being cut off at 8 steps before finishing, so the budget is binding on the outcome rather than the agent's competence. Doubling it separates the two.
- **gen 4** (reverted): Full retention did not stop the context loss, so the problem is finding the relevant earlier fact, not storing it. Retrieval over 6 relevant items targets that directly.

## Analysis and Ladder Status

- **Complete generation execution**: All 4 generations were fully evaluated (16 tasks per generation &times; 4 generations = 64 live generation calls + 48 baseline replicate calls + 16 generation-0 root calls = 128 total calls). None were skipped or truncated.
- **Why mutations were reverted**:
  - Gen 1 (system prompt guidance): reduced score from 0.7500 to 0.6000 (delta -0.1500) -> properly reverted.
  - Gen 2 (strategy change): scored 0.6250 (delta -0.1250) -> properly reverted.
  - Gen 3 (step budget increase): scored 0.7500 (delta +0.0000) -> properly reverted under noise threshold.
  - Gen 4 (memory reconfiguration): scored 0.6875 (delta -0.0625) -> properly reverted.
- **Headroom vs. Mutation Vocabulary Limitation**:
  - `code_math` presents ~35% theoretical headroom (baseline accuracy 0.6458, task-level baseline mean score 0.7500).
  - All four generations targeted `premature_stop`, and none of the four distinct approaches helped (two made scores worse, one was flat, one degraded into sampling noise).
  - This suggests either:
    1. The structural diagnosis rule (`step_count <= max_steps // 2`) misattributes calculation failures as `premature_stop` simply because single-shot calculations finish in 0 tool steps; OR
    2. The current mutation vocabulary (behavioral prompt guidance, strategy mode changes, step budgets, memory architectures) lacks operators capable of improving raw mathematical/algorithmic precision on fixed model weights.
  - This is an honest and informative limitation of the current mutation set: on pure computation without tools, structural agent mutations cannot compensate for model reasoning boundaries, and the selection harness correctly refused to adopt any of the degraded mutations.
