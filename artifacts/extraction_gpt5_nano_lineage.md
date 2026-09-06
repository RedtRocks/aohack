# extraction x a real model: a live run through the engine loop

Primary model: `glm-4-7-flash (via TensorMux)` &middot; fallback: `gpt-5-nano (via OpenAI)` &middot; domain: `extraction` &middot; 14 tasks &middot; **126 real API calls made** (126 primary / 0 fallback)

A weak model was chosen deliberately, not as a fallback: a strong baseline agent leaves the loop no headroom, so every mutation lands in the noise. A weak model gives a low baseline with real room to improve, and it is cheap and fast enough to run several times per task so run-to-run variance means something.

**Baseline synthesizer: `naive`.** We start from a naive baseline so that improvement is measurable. Generation zero is a bare system prompt (the goal string, verbatim, and nothing else -- no tool listing, no worked examples, no 'how to work' scaffolding), single_shot strategy, memory kind none, and a minimal one-step stopping condition. This is a deliberate experimental choice, not the strongest agent we could build: TemplateSynthesizer's own default prompt already tells the model to check tool errors and stop only once complete, which leaves little headroom for the loop's early mutations to visibly improve on. This baseline is weaker on purpose.

## Noise floor

The root spec was run through the real suite 3 independent times before any mutation, to measure how much `mean_score` -- the metric the loop's own accept/revert decision is made on -- moves on its own, from model sampling alone.

- replicate `mean_score` values: [0.9278911428571428, 0.9278911428571428, 0.9136054285714286]
- population variance: 0.000045
- population std (the noise floor): 0.006734

Any lineage delta below this std is flagged `within_noise_floor` below and must not be read as a real improvement or regression.

## Baseline vs. final (accuracy always paired with cost)

| | accuracy | reliability (variance) | cost (mean tokens/run) | speed (mean seconds/run) |
|---|---|---|---|---|
| baseline (`extraction-gpt5-nano`) | 0.8571 (n=42) | 0.0317 (n=14) | 493.3810 (n=42) | 3.3072s (n=42) |
| final (`extraction-gpt5-nano-g1`) | 0.8571 (n=42) | 0.0317 (n=14) | 846.2857 (n=42) | 4.0609s (n=42) |

Both rows are computed by the real, frozen `agent_engineer.evaluation.Evaluator` harness, `repeats=3`, over the exact same recorded trajectories the noise-floor measurement used -- not hand-computed here.

## The real lineage, verbatim

| gen | dominant cause | mutation | before | after | delta | decision | within noise floor |
|---|---|---|---|---|---|---|---|
| 1 | output_format_violation | system_prompt_rewrite (system_prompt) | 0.9279 | 0.9517 | +0.0238 | accepted | no |
| 2 | output_format_violation | strategy_changed (strategy) | 0.9517 | 0.9398 | -0.0119 | reverted | no |

## Mutation-kind distribution

- `system_prompt_rewrite`: 1
- `strategy_changed`: 1

Rationale behind each proposed mutation:

- **gen 1** (accepted): The dominant cause is output_format_violation. Adding explicit behavioural guidance for it is the smallest edit that can change the agent's conduct at the step where it fails.
- **gen 2** (reverted): Prompt-level guidance did not fix output_format_violation, so the loop shape is the constraint: react gives the agent an explicit step for the deliberation it is currently skipping.
