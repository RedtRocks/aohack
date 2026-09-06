# Limitations

Honest accounting of what this build has actually measured. Read this before
citing any number this CLI prints, or any chart built from one.

## Live-model status: two attempts, neither landed a real number yet

This section has changed more than once already and may change again -- read
it as a status log, not a one-time verdict.

**This CLI's own attempt.** As reported through this session's orchestrator,
API keys were supplied directly and a live run against GLM 4.7 Flash (with a
smaller "nano" model as fallback) was said to be in progress against the real
`code_math` domain, run separately from this CLI. **No real numbers from that
run have landed in this repository as of this writing.** Every number this
CLI itself has printed, in every run it has done so far, still comes from a
scripted backend (`agent_engineer.cli.ScriptedBackend`), on all three
domains, with no exception. `agent_engineer.cli.AnthropicBackend` exists in
the source and is reachable via `--backend anthropic`, but it has not been
exercised through this CLI, and no run through it is reported here.

**A second, independent attempt, in a different session working the same
project.** That session was first asked to search for a reachable Anthropic
key (none was reachable), then explicitly asked to use GPT-5 nano via
`OPENAI_API_KEY` (also not reachable in that session's environment), then
told to stand down on the real-model attempt entirely. It stopped rather than
simulate a result either time. What it left behind, staged and tested but
never executed against a real key: `experiments/gpt5_nano_backend.py` (a
`ModelBackend` for GPT-5 nano over the OpenAI-compatible API) and
`experiments/run_real_model_gate.py` (the driver that would wire it to
`agent_engineer.loop.run_loop` against real `code_math` and measure a real
noise floor before comparing any delta against it). `tests/test_gpt5_nano_backend.py`
covers the missing-key path, response parsing, and retries -- with no network
calls anywhere in that test file, confirmed by reading it. `README.md`
documents the exact reproduction command (`OPENAI_API_KEY=sk-...
python experiments/run_real_model_gate.py`), which someone with a working key
can run to produce `artifacts/code_math_gpt5_nano_lineage.{json,md}`. As of
this writing that command has not been run in either session.

So: **as of this writing, no live-model number exists anywhere in this
repository, on any domain, from either attempt.** Once one lands -- from
either session -- this section will say, per domain, which numbers came from
a live model and which still come from a scripted backend, because a document
that says "live" once at the top and stops distinguishing after that is not
honest about a partial result. If a live run fails, times out, or is cut
short, this section says that plainly instead: a run that didn't finish is
not evidence either way, and is not something to round up.

**What the scripted runs DO establish, and it is a real result:**

- The five stages -- synthesize, evaluate, diagnose, mutate, select -- integrate
  correctly through the frozen contracts (`AgentSpec`, `Trajectory`, `Diagnosis`,
  `Mutation`) against three real domains (`code_math`, `extraction`,
  `api_orchestration`), not just against a test double built for the engine's
  own test suite.
- Before-numbers and deltas propagate correctly end to end: every
  `GenerationRecord` carries the exact `before`/`after`/`delta` the selection
  policy decided on, and the CLI's lineage view and results table render
  those same numbers without recomputing or rounding away the connection
  between them.
- Accept/reject/revert behaves correctly across a mixed lineage: this build
  has been run against a lineage that includes **accepted** generations with
  varied, non-uniform deltas (`code_math`: +0.125, +0.375, +0.0625 in one run)
  and **reverted** generations at zero delta (`extraction`,
  `api_orchestration`, where the scripted backend has no reference answer to
  give and correctly refuses rather than guesses). The lineage view was
  designed against this mixed shape from the start -- not against a single
  run of identical accepts -- and renders REVERTED, zero-delta, and
  non-monotonic sequences the same way it renders a clean accept: labeled,
  numbered, and diffed, never hidden or smoothed over.

**What the scripted runs do NOT establish, and no output here should imply
otherwise:**

- That an LLM agent's performance improved, from anything printed by this
  CLI. No LLM agent has run through this CLI as of this writing. A scripted
  backend answering a fixed hash of a task id correctly is not a model getting
  better at a task; it is a stand-in confirming the plumbing that would carry
  a model's improvement, if there were one, without inventing the improvement
  itself. (A live run against a real model is in progress elsewhere, on
  `code_math`, outside this CLI -- see the status note above for what has and
  has not landed here.)
- Whether the mutations the ladder proposes (system-prompt rewrites, strategy
  changes, memory reconfiguration, budget adjustments) would change a real
  model's behavior in the direction their rationale claims.
- Whether the heuristic diagnoser's structural rules (budget exhaustion, tool
  misuse, repeated calls, premature stop) correctly attribute a *real* agent's
  failures, as opposed to a scripted backend's engineered ones.
- Whether the four measured metrics (accuracy, reliability, cost, speed) behave
  the way this document assumes once latency and token variance are real
  instead of fixed per call.

If a chart or table produced by this CLI is shown to anyone, the honest
caption is "the lineage machinery accepts, rejects, and reverts correctly
end to end on a scripted backend" -- not "the agent got better."

A second scripted-backend suite, built separately to specifically exercise
rejected mutations, negative deltas, and a delta smaller than run-to-run
variance, has landed: `tests/test_code_math_loop_decision_stage.py`, tests
against the real `code_math` domain and the unmodified engine:

- a mutation that measures worse than its parent is rejected, with an
  assertion that `report.final_spec` is actually the parent spec, not just a
  decision flag saying so;
- a real reliability variance, measured through the frozen
  `agent_engineer.evaluation.Evaluator` (`repeats=3`, not asserted by fiat) on
  a probe backend with one attempt-flaky task -- population variance
  `0.013889` (noise floor, its standard deviation, `0.117851`) -- used to show
  that a true delta smaller than that noise floor gets reverted rather than
  accepted (see below for how that stopped depending on the caller
  remembering to configure it);
- a mixed lineage -- one reject, one accept, one reject -- saved as
  `artifacts/scripted_decision_lineage.{json,md}`, labelled `SCRIPTED`
  throughout: gen1 `0.6250 -> 0.1875` (`-0.4375`, reverted), gen2
  `0.6250 -> 0.8125` (`+0.1875`, accepted), gen3 `0.8125 -> 0.8750` (`+0.0625`,
  reverted -- inside the `0.117851` noise floor).

**A gap this found in the engine's own default got fixed, not just written
down.** The first version of the test above showed `run_loop`'s default
`MinimumDeltaPolicy(min_delta=1e-9)` would ACCEPT that `+0.0625` delta,
because `1e-9` has no notion of measurement noise and any positive number
clears it. Rather than leave that as a documented hole, `agent_engineer/loop.py`
was changed so that when a caller does not supply their own
`selection_policy`, `run_loop` now measures a real reliability variance for
the root spec through the `Evaluator` harness (`repeats=3`) before running any
generation, and uses the resulting standard deviation as the keep threshold
for the whole lineage automatically -- see `run_loop`'s own docstring and
`LineageReport.noise_floor`. `tests/test_code_math_loop_decision_stage.py::test_the_default_reverts_a_delta_smaller_than_the_measured_noise_floor`
is the regression test; a companion test confirms the old, permissive
behavior is still reachable for a caller who deliberately passes their own
`selection_policy` and wants it. The CLI surfaces the measured threshold in
its lineage summary (`report.noise_floor`) rather than leaving it invisible.
This is a stronger claim than "we noted the gap": the loop found its own
default threshold was too permissive to survive its own reliability
measurement, and that got fixed.

This CLI's own rendering (`agent_engineer/cli.py`, `tests/test_cli.py`) was
built and tested against the same mixed shape independently -- accepted,
reverted, positive, negative, and zero delta together -- rather than against
a single clean climb, so no rework was needed here once this landed; the two
were developed in parallel and agree. This section is the one place in the
repo that speaks to the no-live-model limitation; a second, competing
writeup of the same fact should not exist alongside it.

## The "domain-agnostic" claim, and what it rests on

The engine (`agent_engineer/loop.py`, `agent_engineer/stages/`) and the CLI
(`agent_engineer/cli.py`) contain no domain name, no domain-specific branch,
and no import from `agent_engineer.domains.*` beyond the registry
(`get_suite`, `get_evaluator`, `DOMAIN_NAMES`). That claim is verified
mechanically: `tests/test_cli.py::test_run_domain_completes_for_every_registered_domain`
is parametrized over the live `DOMAIN_NAMES` tuple and asserts the loop
completes for each one without a single per-domain code path.

What that claim does **not** cover:

- **Tools.** The CLI wires `NullToolRuntime`, which exposes no tools to any
  spec. `api_orchestration`'s tasks are built around dependent tool chains
  (a `SimulatedSupportAPI` with `find_customer` / `list_orders` / `get_order`
  / etc.); this CLI cannot exercise that chain at all, tool-free or otherwise,
  because building a `ToolRuntime` that calls a specific domain's API would be
  exactly the per-domain wiring the brief rules out. A domain-agnostic CLI and
  a domain that needs bespoke tools are in genuine tension; this build resolves
  it by staying tool-free everywhere, which is honest but means
  `api_orchestration` is currently unreachable through its own strong suit.
- **A fourth domain would need to fit the same two-function contract**
  (`get_suite() -> DomainSuite`, `get_evaluator() -> TaskEvaluator`) documented
  in `agent_engineer/evaluation/INTERFACE.md`. Nothing about the CLI enforces
  that a new domain package actually satisfies the contract at import time
  beyond what `agent_engineer.domains.get_suite`/`get_evaluator` already do; a
  malformed domain fails the same way a malformed domain always would, with
  no CLI-specific guardrail added or needed.

## There is no LLM judge here, so there is nothing to validate

All three domain evaluators were checked directly, and each grades a
trajectory deterministically:

- **`code_math`** (`agent_engineer/domains/code_math/evaluator.py`): settles
  every answer by executing the submitted code against fixed test cases, or by
  exact/numeric comparison against a reference value. No model call anywhere
  in the file.
- **`extraction`** (`agent_engineer/domains/extraction/evaluator.py`): its own
  docstring states it plainly -- "deterministic end to end: no model call, no
  network, no clock." Grading is field-by-field normalized string/number
  comparison against frozen ground truth.
- **`api_orchestration`** (`agent_engineer/domains/api_orchestration/evaluator.py`):
  splits grading into an answer half (regex/normalized-text matching against
  rubric facts) and a chain half (checking the actual tool calls made, with
  the right threaded arguments). Both halves are pattern matching over a
  simulated, deterministic API; no model call.

So: **there is no LLM-as-judge in this build, on any of the three domains.**
That is a stronger and cheaper claim than an inter-rater kappa score would be,
and it is the claim this document makes instead of one. If a future domain
adds a model-graded evaluator, that evaluator needs its own validation pass
(agreement against a human-labeled sample) before its numbers are trusted --
nothing in the engine or the CLI performs that validation today, because
nothing here currently needs it.

## The measurement harness's guards, and what they still don't do

`agent_engineer/evaluation/report.py` refuses to render an improvement without
its before number, and refuses accuracy without its paired cost; the CLI
routes every results table through it rather than formatting numbers directly
(`agent_engineer/cli.py::run_domain`, `render_report`). Those guards are
enforced by the shapes of the functions, not by convention, and this build
never routes around them.

What they don't do: they don't make a metric *meaningful* by construction --
`reliability` is undefined below three repeat runs per task
(`MIN_RUNS_FOR_RELIABILITY = 3`), and every result the CLI has produced so far
used a scripted backend whose outcomes are hash-deterministic per task, so its
reliability metric is trivially `0.0000` (identical pass/fail every repeat),
never a real measure of a real model's run-to-run variance. That number is
real, and it is undefined-safe, but on `--backend scripted` runs it is not
informative until a real backend is behind it.
