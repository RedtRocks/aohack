# Real-model gate crossing: code_math x GPT-5 nano

**Status: not yet run.** No `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`) has been
reachable in the environments this work was done in, and no run was simulated
in its place. Everything below is staged and tested (`tests/test_gpt5_nano_backend.py`
covers the backend itself, without any network call) so that anyone with a key
can produce the real artifact with the one command below.
`tests/test_code_math_loop_decision_stage.py` and
`generate_scripted_decision_lineage.py` cover the *decision logic* (accept,
reject, revert, noise floor) with scripted backends in the meantime -- see
their docstrings for what that does and does not establish.

`tests/test_code_math_loop_end_to_end.py` proves the engine's five stages wire
together: mutations propose, get measured, get accepted or reverted, and the
lineage carries before/after/delta. It does that against a scripted backend
whose competence is `base + step * generation` -- deliberately so, as a fast,
deterministic regression test -- but that means the improvement in that test
is arithmetic in the test double, not something the engine's mutations caused.

This directory answers the harder question: does the loop actually improve a
*real* agent? `run_real_model_gate.py` wires `agent_engineer.loop.run_loop`,
unmodified, to a real model (GPT-5 nano, via `gpt5_nano_backend.py`) against
the real `agent_engineer.domains.code_math` suite and evaluator, and saves the
verbatim result to `../artifacts/code_math_gpt5_nano_lineage.{json,md}`.

## Why GPT-5 nano, specifically

Not a fallback -- a deliberate experimental choice. A strong baseline agent
leaves the loop no headroom: every task already passes, so every mutation
measures a flat delta and the loop has nothing to demonstrate. A weak model
gives a low baseline with real room to move, and it is cheap and fast enough
to run several times per task, which is what makes a noise floor ("how much
does the score move on its own, from sampling alone, with nothing changed")
measurable at all.

## Running it

Requires `OPENAI_API_KEY` in the environment and the `requests` package
installed (it is not a declared dependency of the `agent_engineer` package
itself -- this is a one-off experiment script, not engine or domain code).

```
OPENAI_API_KEY=... python experiments/run_real_model_gate.py
python experiments/render_lineage_markdown.py
```

The first command makes on the order of 150-200 real API calls (three
baseline replicates + the loop's own generations + three final-spec
replicates, each over the 16-task suite) and writes the JSON artifact. The
second renders it to Markdown.

## What the artifact reports, and why each number is there

- **Noise floor**: the root spec run three independent times before any
  mutation, giving three `mean_score` samples and their variance -- the same
  metric the loop's own accept/revert decision is made on. Any lineage delta
  smaller than this std is flagged, never reported as a real improvement.
- **Baseline vs. final, accuracy paired with cost**: both computed by the
  real, frozen `agent_engineer.evaluation.Evaluator` (`repeats=3`), replayed
  over the exact trajectories the noise-floor measurement already collected --
  not hand-computed here. Accuracy is never reported without its cost, because
  more retries always buys accuracy.
- **The lineage itself**: every mutation's dominant cause, kind, before,
  after, delta, and accept/revert decision, verbatim from `run_loop`.
