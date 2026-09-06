# agent-engineer

A domain-agnostic agent-improvement loop: synthesize an agent spec, evaluate
it against a task suite, diagnose why it failed, propose one targeted
mutation, and keep or revert it based on a measured delta. See
`agent_engineer/schemas.py` for the four frozen contracts every stage shares,
and `agent_engineer/evaluation/INTERFACE.md` for the measurement contract a
domain package conforms to.

## Running the tests

```
pip install -e .[dev]
pytest
```

## Reproducing a real-model run

`experiments/` holds a `ModelBackend` for real GPT-5 nano
(`experiments/gpt5_nano_backend.py`) and a driver
(`experiments/run_real_model_gate.py`) that wires the unmodified
`agent_engineer.loop.run_loop` to it against the real `code_math` domain,
measuring a real run-to-run noise floor before comparing any lineage delta
against it. It is unused by default -- nothing in the main package or test
suite imports it -- and it makes real, billed API calls, so it is not part of
`pytest`. To reproduce it:

```
pip install requests
OPENAI_API_KEY=sk-... python experiments/run_real_model_gate.py
python experiments/render_lineage_markdown.py
```

This writes `artifacts/code_math_gpt5_nano_lineage.{json,md}`. See
`experiments/README.md` for what each number in that artifact means and why
GPT-5 nano specifically. **As of this writing this has not been run**: no
API key has been available in the environments this work was done in, and no
run was simulated in its place. `artifacts/scripted_decision_lineage.{json,md}`
is a separate, clearly-labelled scripted artifact (see
`experiments/generate_scripted_decision_lineage.py`) showing the loop's
accept/reject/revert decision exercised end to end -- a rejected mutation
with a negative delta, an accepted one, and a rejected one whose delta sat
inside a measured reliability variance -- with no live model involved.

See `LIMITATIONS.md` for what the current test suite does and does not
establish about a real model's performance.
