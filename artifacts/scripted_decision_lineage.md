# code_math keep-or-revert, exercised end to end (SCRIPTED)

**SCRIPTED -- no live model was used to produce this lineage.** _TieredBackend: solves a fixed, chosen number of the 16 code_math tasks correctly per generation, read off the spec id. Not a model.

Domain: `code_math` &middot; 16 tasks &middot; selection policy: `MinimumDeltaPolicy(min_delta=0.117851)  # the measured noise floor`

## Noise floor

agent_engineer.evaluation.Evaluator, repeats=3, on a probe backend with one attempt-flaky task (population variance of one flip in three attempts is 2/9, spread over the suite).

- reliability (variance): `0.013889`
- noise floor (std): `0.117851`

## The lineage

| gen | dominant cause | mutation | before | after | delta | decision |
|---|---|---|---|---|---|---|
| 1 | premature_stop | system_prompt_rewrite (system_prompt) | 0.6250 | 0.1875 | -0.4375 | reverted |
| 2 | premature_stop | strategy_changed (strategy) | 0.6250 | 0.8125 | +0.1875 | accepted |
| 3 | premature_stop | stopping_adjusted (stopping.max_steps) | 0.8125 | 0.8750 | +0.0625 | reverted |

Root spec: `mixed-lineage-agent` &rarr; final spec: `mixed-lineage-agent-g2`

Reasons, verbatim from the selection stage:

- **gen 1** (reverted): measured 0.6250 -> 0.1875 (delta -0.4375), which does not clear the 0.117851 keep threshold; reverting to the parent spec
- **gen 2** (accepted): measured 0.6250 -> 0.8125 (delta +0.1875), which clears the 0.117851 keep threshold
- **gen 3** (reverted): measured 0.8125 -> 0.8750 (delta +0.0625), which does not clear the 0.117851 keep threshold; reverting to the parent spec
