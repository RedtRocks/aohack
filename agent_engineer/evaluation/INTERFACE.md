# The evaluator interface

Owned by the measurement agent. Everything measured -- accuracy, reliability,
cost, speed -- is computed by this one module, off the same runs. If accuracy
and cost were measured by different code written by different people the numbers
would stop being comparable and nobody would find out until the demo.

Import it, do not reimplement it:

```python
from agent_engineer.evaluation import Evaluator, DomainSuite, TaskSpec
```

This interface will not change under you. If you need a field on the four frozen
schemas, route the request through the orchestrator to the schemas owner; do not
edit `agent_engineer/schemas.py`.

## Domain agents supply exactly two things

**1. A `DomainSuite`.** Export it as a module-level `get_suite() -> DomainSuite`
so the engine can collect all three domains uniformly.

```python
TaskSpec(task_id=..., domain=..., prompt=..., expected=None, metadata={})
DomainSuite(domain=..., tasks=(task, ...))
```

Every `task.domain` must equal `suite.domain`, and `task_id`s must be unique
within the suite. `metadata` is opaque to the evaluator -- put domain fixtures
there.

**2. A `TaskEvaluator`** -- a plain callable, exported as
`get_evaluator() -> TaskEvaluator`:

```python
(task: TaskSpec, trajectory: Trajectory) -> EvaluatorVerdict
```

Grade only. Do **not** compute accuracy, cost, speed, or reliability, and do not
aggregate or average anything -- counting is the evaluator's job. If a
Trajectory already carries a verdict, the evaluator uses it and never calls you.

## The engine supplies exactly one thing

A `TaskRunner` callable:

```python
(spec: AgentSpec, task: TaskSpec, attempt: int) -> Trajectory
```

Called `repeats` times per task, with `attempt` counting from 0. It may raise: a
raised run is recorded as a **failure** and stays in the denominator. Never
swallow an error and skip the task -- that inflates accuracy.

Set `Trajectory.tokens` and `Trajectory.elapsed_seconds`. Cost and speed are read
from those fields and nowhere else.

## The whole integration surface

```python
from agent_engineer.evaluation import Evaluator

evaluator = Evaluator(suites, evaluators={domain: task_evaluator}, repeats=3)
report = evaluator.run_iteration(iteration_index, spec, runner)

report.domain("extraction").accuracy   # -> Metric
report.average("accuracy")             # cross-domain mean over DEFINED metrics only
```

`Metric(kind, value: float | None, sample_size, reason)`. A `value` of `None`
means **undefined**: never render it as `0`, never average it in.
`Metric.require()` raises rather than hand back a value that is not there.

## The three rules, as implemented

1. **Accuracy is over all tasks, and a refusal is a failure.** The denominator is
   `len(tasks) * repeats`, fixed before the first run. A refusal (no final
   answer, a blank one, or a stock decline phrase), a runner exception, and a run
   nobody graded all count as failures. An agent that declines the hard tasks
   cannot score beautifully on the rest.
2. **An empty denominator is undefined, not zero.** A domain where nothing ran
   reports `Metric(value=None, reason=...)` and is excluded from every average,
   rather than folding a `0.0` in and quietly dragging the aggregate down.
3. **Reliability needs at least three runs per task and is reported as
   variance.** `repeats < 3` is a constructor error. Reliability is the
   population variance of the pass indicator across runs -- never a mean.

## Reporting rules

Enforced by the rendering layer, not left to the caller's discipline:

- No improvement is rendered without its before number.
- No accuracy figure is rendered without its paired cost. More retries always
  buys accuracy, so accuracy alone is not a result.
