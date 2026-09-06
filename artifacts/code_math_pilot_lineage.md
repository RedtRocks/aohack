# code_math x a real model: a live run through the engine loop

Primary model: `glm-4-7-flash (via TensorMux)` &middot; fallback: `gpt-5-nano (via OpenAI)` &middot; domain: `code_math` &middot; 3 tasks &middot; **12 real API calls made** (12 primary / 0 fallback)

A weak model was chosen deliberately, not as a fallback: a strong baseline agent leaves the loop no headroom, so every mutation lands in the noise. A weak model gives a low baseline with real room to improve, and it is cheap and fast enough to run several times per task so run-to-run variance means something.

## Noise floor

The root spec was run through the real suite 3 independent times before any mutation, to measure how much `mean_score` -- the metric the loop's own accept/revert decision is made on -- moves on its own, from model sampling alone.

- replicate `mean_score` values: [1.0, 1.0, 1.0]
- population variance: 0.000000
- population std (the noise floor): 0.000000

Any lineage delta below this std is flagged `within_noise_floor` below and must not be read as a real improvement or regression.

## Baseline vs. final (accuracy always paired with cost)

| | accuracy | reliability (variance) | cost (mean tokens/run) | speed (mean seconds/run) |
|---|---|---|---|---|
| baseline (`code-math-pilot`) | 1.0000 (n=9) | 0.0000 (n=3) | 343.1111 (n=9) | 3.9337s (n=9) |
| final (`code-math-pilot`) | 1.0000 (n=9) | 0.0000 (n=3) | 343.1111 (n=9) | 3.9337s (n=9) |

Both rows are computed by the real, frozen `agent_engineer.evaluation.Evaluator` harness, `repeats=3`, over the exact same recorded trajectories the noise-floor measurement used -- not hand-computed here.

## The real lineage, verbatim

| gen | dominant cause | mutation | before | after | delta | decision | within noise floor |
|---|---|---|---|---|---|---|---|

Rationale behind each proposed mutation:

