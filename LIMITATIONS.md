# Limitations

Honest accounting of what this build has actually measured, versus what it is
designed to measure once the pieces currently missing land. Read this before
citing any number this CLI prints.

## No real model has run through this yet

Every number in this repo's test suite, and every number the CLI prints by
default (`--backend scripted`), comes from `agent_engineer.cli.ScriptedBackend`
-- a deterministic stand-in, not a model. It answers a task correctly only if
that task carries a generic reference answer (`TaskSpec.expected`) and admits
it based on a fixed hash of the task id and the generation tier, not on any
actual reasoning. It never reads a prompt, calls a tool, or does anything an
agent does.

`agent_engineer.cli.AnthropicBackend` exists and is wired into `--backend
anthropic`, but it has not been run: this environment has no
`ANTHROPIC_API_KEY` configured, and nothing below should be read as if a real
model had exercised this path. Until it does, the following remain unverified
against a real model:

- Whether the mutations the ladder proposes (system-prompt rewrites, strategy
  changes, memory reconfiguration, budget adjustments) actually change a real
  model's behavior in the direction the rationale claims.
- Whether the heuristic diagnoser's structural rules (budget exhaustion, tool
  misuse, repeated calls, premature stop) correctly attribute a *real* agent's
  failures, as opposed to a scripted backend's engineered ones.
- Whether the four measured metrics (accuracy, reliability, cost, speed) look
  the way this document assumes once latency and token variance are real
  instead of fixed per call.

**What has been measured**, and is safe to cite: the *engine's own control
flow* -- synthesize, evaluate, diagnose, mutate, select, repeat -- runs
correctly against three real domains (`code_math`, `extraction`,
`api_orchestration`), and the CLI built on top of it streams a legible lineage
and an honest results table for whatever the scripted backend produces on each
one. `code_math` shows repeated **accepted** generations with varied,
non-uniform deltas (+0.125, +0.375, +0.0625 in one run) because most of its
tasks carry a reference answer the scripted backend can hit. `extraction` and
`api_orchestration` show **reverted** generations at zero delta, because most
of their tasks either have no single reference answer or need free-form
prose/tool chains this backend does not attempt -- which is the honest result
of a backend that refuses rather than guesses, not a defect in the CLI.

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
