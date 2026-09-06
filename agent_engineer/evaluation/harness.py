"""The evaluator harness: the interface the engine and the domain agents conform to.

A domain agent supplies two things and nothing else:

* a :class:`DomainSuite` -- a domain name and its :class:`TaskSpec` list;
* a :class:`TaskEvaluator` -- ``(TaskSpec, Trajectory) -> EvaluatorVerdict``.

The engine supplies one thing:

* a :class:`TaskRunner` -- ``(AgentSpec, TaskSpec, attempt) -> Trajectory``.

:class:`Evaluator` owns everything between them: repeat runs, the four metrics,
the denominators, and the per-iteration report. Domain code never computes a
metric, and the engine never computes one either -- if accuracy and cost were
measured by different code they would stop being comparable.

Three rules are enforced here and each is pinned by a test:

1. **Accuracy is over all tasks, never over attempted tasks, and a refusal is a
   failure.** The denominator is ``len(tasks) * repeats``, fixed before the
   first run. A task that raised, never ran, produced no answer, or was politely
   declined lands in the denominator as a failure. An agent cannot raise its
   score by skipping the hard tasks.
2. **An empty denominator is undefined, not zero.** A domain with no runs at all
   reports ``Metric(value=None)`` and is excluded from every average.
3. **Reliability needs at least three runs per task and is reported as
   variance**, never as a mean.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Annotated, Protocol

from pydantic import BaseModel, ConfigDict, Field

from agent_engineer.evaluation.metrics import (
    Metric,
    MetricKind,
    mean_of_defined,
    population_variance,
    undefined,
)
from agent_engineer.schemas import AgentSpec, EvaluatorVerdict, Trajectory

__all__ = [
    "MIN_RUNS_FOR_RELIABILITY",
    "REFUSAL_MARKERS",
    "TaskSpec",
    "DomainSuite",
    "TaskRunner",
    "TaskEvaluator",
    "RunOutcome",
    "TaskResult",
    "DomainReport",
    "IterationReport",
    "Evaluator",
    "is_refusal",
]

MIN_RUNS_FOR_RELIABILITY = 3
"""Reliability is a spread, and a spread needs at least three points to mean anything."""

NonEmptyStr = Annotated[str, Field(min_length=1)]

REFUSAL_MARKERS: tuple[str, ...] = (
    "i can't help",
    "i cannot help",
    "i can't assist",
    "i cannot assist",
    "i'm unable to",
    "i am unable to",
    "i won't be able",
    "i will not be able",
    "i don't have the ability",
    "unable to complete this task",
    "cannot complete this task",
)
"""Lowercased substrings that mark a declined task. Declining is a failure, not an abstention."""


# --------------------------------------------------------------------------- #
# What the domain agents supply
# --------------------------------------------------------------------------- #


class TaskSpec(BaseModel):
    """One task in a domain suite. The unit accuracy is computed over.

    ``task_id`` must match the ``task_id`` on the Trajectory produced for it, so
    runs can be attributed back to tasks.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: NonEmptyStr
    domain: NonEmptyStr
    prompt: NonEmptyStr
    expected: str | None = Field(
        default=None, description="Reference answer, when the domain has one."
    )
    metadata: dict[str, object] = Field(
        default_factory=dict, description="Domain-specific fixtures. Opaque to the evaluator."
    )


class DomainSuite(BaseModel):
    """A domain's tasks. Delivered by the domain agent, consumed by :class:`Evaluator`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    domain: NonEmptyStr
    tasks: tuple[TaskSpec, ...] = ()

    def model_post_init(self, _context: object) -> None:
        seen: set[str] = set()
        for task in self.tasks:
            if task.domain != self.domain:
                raise ValueError(
                    f"task {task.task_id!r} declares domain {task.domain!r} "
                    f"but sits in suite {self.domain!r}"
                )
            if task.task_id in seen:
                raise ValueError(f"duplicate task_id {task.task_id!r} in suite {self.domain!r}")
            seen.add(task.task_id)


class TaskRunner(Protocol):
    """Engine-supplied. Runs one spec against one task once and returns its Trajectory.

    Called ``repeats`` times per task with ``attempt`` counting from 0. May
    raise: a raised run is recorded as a failure, never dropped from the
    denominator.
    """

    def __call__(self, spec: AgentSpec, task: TaskSpec, attempt: int) -> Trajectory: ...


class TaskEvaluator(Protocol):
    """Domain-supplied. Judges one trajectory against its task.

    Used only when the Trajectory does not already carry a verdict. Grading is
    the domain's business; counting is not.
    """

    def __call__(self, task: TaskSpec, trajectory: Trajectory) -> EvaluatorVerdict: ...


# --------------------------------------------------------------------------- #
# What the evaluator produces
# --------------------------------------------------------------------------- #


class RunOutcome(BaseModel):
    """One attempt at one task: did it pass, and what did it cost."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: NonEmptyStr
    attempt: int = Field(ge=0)
    passed: bool
    refused: bool = Field(default=False, description="Declined the task. Always counts as failure.")
    errored: bool = Field(default=False, description="The runner raised. Always counts as failure.")
    total_tokens: int = Field(default=0, ge=0)
    elapsed_seconds: float = Field(default=0.0, ge=0.0)
    trajectory_id: str | None = None
    detail: str = ""

    def model_post_init(self, _context: object) -> None:
        if self.passed and (self.refused or self.errored):
            raise ValueError("a refused or errored run cannot be recorded as passed")


class TaskResult(BaseModel):
    """Every attempt at one task, plus that task's reliability."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: NonEmptyStr
    outcomes: tuple[RunOutcome, ...]

    @property
    def success_rate(self) -> float:
        """Passes over *all* attempts. Refusals and errors sit in the denominator."""
        return sum(1 for outcome in self.outcomes if outcome.passed) / len(self.outcomes)

    def variance(self) -> Metric:
        """Variance of the pass indicator across attempts. Undefined below three runs."""
        if len(self.outcomes) < MIN_RUNS_FOR_RELIABILITY:
            return undefined(
                "reliability",
                f"task {self.task_id!r} has {len(self.outcomes)} run(s); "
                f"reliability needs at least {MIN_RUNS_FOR_RELIABILITY}",
            )
        indicators = [1.0 if outcome.passed else 0.0 for outcome in self.outcomes]
        return Metric(
            kind="reliability",
            value=population_variance(indicators),
            sample_size=len(indicators),
        )


class DomainReport(BaseModel):
    """The four metrics for one domain at one iteration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    domain: NonEmptyStr
    iteration: int = Field(ge=0)
    spec_id: NonEmptyStr
    task_count: int = Field(ge=0)
    repeats: int = Field(ge=0)
    results: tuple[TaskResult, ...] = ()
    accuracy: Metric
    reliability: Metric
    cost: Metric
    speed: Metric

    @property
    def metrics(self) -> dict[MetricKind, Metric]:
        return {
            "accuracy": self.accuracy,
            "reliability": self.reliability,
            "cost": self.cost,
            "speed": self.speed,
        }

    @property
    def refusal_count(self) -> int:
        return sum(1 for result in self.results for outcome in result.outcomes if outcome.refused)


class IterationReport(BaseModel):
    """Every domain at one iteration, plus cross-domain averages over defined metrics only."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    iteration: int = Field(ge=0)
    spec_id: NonEmptyStr
    domains: tuple[DomainReport, ...] = ()

    def domain(self, name: str) -> DomainReport:
        for report in self.domains:
            if report.domain == name:
                return report
        raise KeyError(f"no domain {name!r} in iteration {self.iteration}")

    def average(self, kind: MetricKind) -> Metric:
        """Cross-domain mean, excluding domains where the metric is undefined."""
        return mean_of_defined((report.metrics[kind] for report in self.domains), kind=kind)

    @property
    def mean_accuracy(self) -> Metric:
        return self.average("accuracy")

    @property
    def mean_reliability(self) -> Metric:
        return self.average("reliability")

    @property
    def mean_cost(self) -> Metric:
        return self.average("cost")

    @property
    def mean_speed(self) -> Metric:
        return self.average("speed")


# --------------------------------------------------------------------------- #
# Refusal detection
# --------------------------------------------------------------------------- #


def is_refusal(trajectory: Trajectory) -> bool:
    """True when the run declined the task rather than attempting it.

    A run with no final answer at all counts as a refusal, as does one whose
    answer is a stock decline. Either way it is a failure: an agent that ducks
    the hard tasks must not outscore one that tries and misses.
    """
    answer = trajectory.final_answer
    if answer is None or not answer.strip():
        return True
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


# --------------------------------------------------------------------------- #
# The evaluator
# --------------------------------------------------------------------------- #


class Evaluator:
    """Runs a spec over domain suites and computes the four metrics.

    One owner for all four numbers, so accuracy and cost are always measured off
    the same runs and stay comparable across domains and iterations.
    """

    def __init__(
        self,
        suites: Sequence[DomainSuite],
        *,
        evaluators: dict[str, TaskEvaluator] | None = None,
        repeats: int = MIN_RUNS_FOR_RELIABILITY,
    ) -> None:
        if repeats < MIN_RUNS_FOR_RELIABILITY:
            raise ValueError(
                f"repeats={repeats} is too few: reliability is a variance and needs at "
                f"least {MIN_RUNS_FOR_RELIABILITY} runs per task"
            )
        names = [suite.domain for suite in suites]
        if len(names) != len(set(names)):
            raise ValueError("duplicate domain in suites")
        self.suites = tuple(suites)
        self.evaluators = dict(evaluators or {})
        self.repeats = repeats

    def run_iteration(self, iteration: int, spec: AgentSpec, runner: TaskRunner) -> IterationReport:
        """Run every task in every suite ``repeats`` times and report per domain."""
        return IterationReport(
            iteration=iteration,
            spec_id=spec.spec_id,
            domains=tuple(self._run_domain(iteration, spec, suite, runner) for suite in self.suites),
        )

    def _run_domain(
        self, iteration: int, spec: AgentSpec, suite: DomainSuite, runner: TaskRunner
    ) -> DomainReport:
        results = tuple(self._run_task(spec, suite, task, runner) for task in suite.tasks)
        return DomainReport(
            domain=suite.domain,
            iteration=iteration,
            spec_id=spec.spec_id,
            task_count=len(suite.tasks),
            repeats=self.repeats,
            results=results,
            accuracy=self._accuracy(suite, results),
            reliability=self._reliability(suite, results),
            cost=self._mean_over_runs(
                suite, results, kind="cost", pick=lambda outcome: float(outcome.total_tokens)
            ),
            speed=self._mean_over_runs(
                suite, results, kind="speed", pick=lambda outcome: outcome.elapsed_seconds
            ),
        )

    def _run_task(
        self, spec: AgentSpec, suite: DomainSuite, task: TaskSpec, runner: TaskRunner
    ) -> TaskResult:
        outcomes = tuple(
            self._run_once(spec, suite, task, runner, attempt) for attempt in range(self.repeats)
        )
        return TaskResult(task_id=task.task_id, outcomes=outcomes)

    def _run_once(
        self,
        spec: AgentSpec,
        suite: DomainSuite,
        task: TaskSpec,
        runner: TaskRunner,
        attempt: int,
    ) -> RunOutcome:
        started = time.monotonic()
        try:
            trajectory = runner(spec, task, attempt)
        except Exception as error:  # a crashed run is a failed run, not a missing one
            return RunOutcome(
                task_id=task.task_id,
                attempt=attempt,
                passed=False,
                errored=True,
                elapsed_seconds=time.monotonic() - started,
                detail=f"{type(error).__name__}: {error}",
            )

        if trajectory.task_id != task.task_id:
            # Grading this would credit one task's run to another. Fail it loudly
            # in the detail rather than quietly mis-attributing a pass.
            return RunOutcome(
                task_id=task.task_id,
                attempt=attempt,
                passed=False,
                errored=True,
                elapsed_seconds=trajectory.elapsed_seconds,
                trajectory_id=trajectory.trajectory_id,
                detail=(
                    f"runner returned a trajectory for task {trajectory.task_id!r} "
                    f"when asked for {task.task_id!r}"
                ),
            )

        refused = is_refusal(trajectory)
        verdict = trajectory.verdict
        if verdict is None:
            judge = self.evaluators.get(suite.domain)
            verdict = judge(task, trajectory) if judge is not None else None
        # No verdict means nobody confirmed a pass, so it is not one.
        passed = verdict is not None and verdict.passed and not refused
        return RunOutcome(
            task_id=task.task_id,
            attempt=attempt,
            passed=passed,
            refused=refused,
            total_tokens=trajectory.tokens.total_tokens,
            elapsed_seconds=trajectory.elapsed_seconds,
            trajectory_id=trajectory.trajectory_id,
            detail="refused" if refused else (verdict.rationale if verdict else "unevaluated"),
        )

    # -- metric computation ------------------------------------------------- #

    def _expected_runs(self, suite: DomainSuite) -> int:
        """The denominator, fixed before any run. Skipping a task cannot shrink it."""
        return len(suite.tasks) * self.repeats

    def _accuracy(self, suite: DomainSuite, results: tuple[TaskResult, ...]) -> Metric:
        expected = self._expected_runs(suite)
        if expected == 0:
            return undefined("accuracy", f"no task ran in domain {suite.domain!r}")
        passed = sum(1 for result in results for outcome in result.outcomes if outcome.passed)
        return Metric(kind="accuracy", value=passed / expected, sample_size=expected)

    def _reliability(self, suite: DomainSuite, results: tuple[TaskResult, ...]) -> Metric:
        if not results:
            return undefined("reliability", f"no task ran in domain {suite.domain!r}")
        return mean_of_defined((result.variance() for result in results), kind="reliability")

    def _mean_over_runs(
        self,
        suite: DomainSuite,
        results: tuple[TaskResult, ...],
        *,
        kind: MetricKind,
        pick: Callable[[RunOutcome], float],
    ) -> Metric:
        values = [pick(outcome) for result in results for outcome in result.outcomes]
        if not values:
            return undefined(kind, f"no task ran in domain {suite.domain!r}")
        return Metric(kind=kind, value=sum(values) / len(values), sample_size=len(values))
