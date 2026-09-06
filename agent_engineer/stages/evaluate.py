"""Stage 2: run an :class:`AgentSpec` against a task set, recording full trajectories.

The recording is complete by construction: every tool call keeps its arguments
and its result (or its error), its own elapsed time and token usage; the run
keeps aggregate tokens, aggregate wall clock, the final answer, and the
evaluator's verdict. Nothing is summarized away, because stage 3 diagnoses from
this record alone.

Stopping is enforced here rather than trusted to the model: the spec's
:class:`StoppingConditions` are checked before every step, and why the loop
halted is preserved in :class:`RunRecord.stop_reason` -- which is what lets
stage 3 tell a budget exhaustion apart from an agent that simply gave up.

Grading happens through a :class:`~agent_engineer.ports.TaskEvaluator`
supplied by the measurement side; this stage calls it once per trajectory and
stores the verdict verbatim, and it is also what :meth:`TrajectoryRunner.as_task_runner`
exposes to the measurement harness as a :class:`~agent_engineer.ports.TaskRunner`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from agent_engineer.ports import (
    AgentAction,
    DomainSuite,
    ModelBackend,
    TaskEvaluator,
    TaskSpec,
    ToolRuntime,
)
from agent_engineer.schemas import AgentSpec, TokenUsage, ToolCall, Trajectory

RESULT_CLIP_CHARS = 4000


class StopReason(str, Enum):
    """Why one run halted."""

    FINAL_ANSWER = "final_answer"
    MAX_STEPS = "max_steps"
    MAX_TOKENS = "max_tokens"
    MAX_WALL_CLOCK = "max_wall_clock"
    STOP_ON_TOOL = "stop_on_tool"
    BACKEND_ERROR = "backend_error"


@dataclass(frozen=True)
class RunRecord:
    """A trajectory plus the loop-level fact of why it stopped."""

    trajectory: Trajectory
    stop_reason: StopReason

    @property
    def budget_exhausted(self) -> bool:
        return self.stop_reason in {
            StopReason.MAX_STEPS,
            StopReason.MAX_TOKENS,
            StopReason.MAX_WALL_CLOCK,
        }


@dataclass(frozen=True)
class EvaluationRun:
    """Every run of one spec over one task set, and the aggregates over them."""

    spec_id: str
    task_set_id: str
    records: tuple[RunRecord, ...]

    @property
    def trajectories(self) -> tuple[Trajectory, ...]:
        return tuple(record.trajectory for record in self.records)

    @property
    def failures(self) -> tuple[Trajectory, ...]:
        return tuple(t for t in self.trajectories if t.failed)

    @property
    def pass_rate(self) -> float:
        """Fraction of evaluated runs that passed. 0.0 over an empty set.

        This is the engine's own bookkeeping for stage 3 and stage 5, over the
        verdicts the measurement evaluator produced -- it is not, itself, a
        metric the engine invents; it never averages anything the evaluator
        did not already score per trajectory.
        """
        evaluated = [t for t in self.trajectories if t.verdict is not None]
        if not evaluated:
            return 0.0
        return sum(1 for t in evaluated if t.verdict.passed) / len(evaluated)

    @property
    def mean_score(self) -> float:
        evaluated = [t for t in self.trajectories if t.verdict is not None]
        if not evaluated:
            return 0.0
        return sum(t.verdict.score for t in evaluated) / len(evaluated)

    def record_for(self, trajectory_id: str) -> RunRecord | None:
        for record in self.records:
            if record.trajectory.trajectory_id == trajectory_id:
                return record
        return None


def _clip(value: object, limit: int = RESULT_CLIP_CHARS) -> object:
    """Keep a tool result storable without letting one huge blob dominate a record."""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"... [{len(value) - limit} more characters]"
    return value


class TrajectoryRunner:
    """Runs specs against tasks and hands each finished run to the evaluator."""

    def __init__(
        self, backend: ModelBackend, tool_runtime: ToolRuntime, evaluator: TaskEvaluator
    ) -> None:
        self._backend = backend
        self._tools = tool_runtime
        self._evaluator = evaluator

    def run_task(
        self, spec: AgentSpec, task: TaskSpec, *, trajectory_id: str | None = None
    ) -> RunRecord:
        """Run one task to completion and return its recorded, evaluated trajectory."""
        exposed = set(spec.tools)
        schemas = tuple(schema for schema in self._tools.schemas() if schema.name in exposed)
        stopping = spec.stopping
        started = time.monotonic()
        calls: list[ToolCall] = []
        prompt_tokens = 0
        completion_tokens = 0
        final_answer: str | None = None
        stop_reason = StopReason.MAX_STEPS

        while True:
            elapsed = time.monotonic() - started
            used_tokens = prompt_tokens + completion_tokens
            if stopping.max_steps is not None and len(calls) >= stopping.max_steps:
                stop_reason = StopReason.MAX_STEPS
                break
            if stopping.max_tokens is not None and used_tokens >= stopping.max_tokens:
                stop_reason = StopReason.MAX_TOKENS
                break
            if (
                stopping.max_wall_clock_seconds is not None
                and elapsed >= stopping.max_wall_clock_seconds
            ):
                stop_reason = StopReason.MAX_WALL_CLOCK
                break

            try:
                action = self._backend.next_action(spec, task, schemas, tuple(calls))
            except Exception as error:  # a backend fault is a run outcome, not a crash
                calls.append(
                    ToolCall(
                        ordinal=len(calls),
                        tool_name="<backend>",
                        args={},
                        error=f"backend raised {type(error).__name__}: {error}",
                    )
                )
                final_answer = None
                stop_reason = StopReason.BACKEND_ERROR
                break

            prompt_tokens += max(0, action.prompt_tokens)
            completion_tokens += max(0, action.completion_tokens)

            if action.is_final:
                final_answer = action.final_answer
                stop_reason = StopReason.FINAL_ANSWER
                break

            calls.append(self._invoke(task, action, ordinal=len(calls)))
            if action.tool_name in set(stopping.stop_on_tools):
                stop_reason = StopReason.STOP_ON_TOOL
                break

        trajectory = Trajectory(
            trajectory_id=trajectory_id or f"{spec.spec_id}::{task.task_id}",
            spec_id=spec.spec_id,
            task_id=task.task_id,
            tool_calls=tuple(calls),
            tokens=TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
            elapsed_seconds=time.monotonic() - started,
            final_answer=final_answer,
        )
        verdict = self._evaluator(task, trajectory)
        trajectory = trajectory.model_copy(update={"verdict": verdict})
        return RunRecord(trajectory=trajectory, stop_reason=stop_reason)

    def _invoke(self, task: TaskSpec, action: AgentAction, *, ordinal: int) -> ToolCall:
        if action.tool_name is None:
            raise ValueError("_invoke requires a tool-calling action")
        call_started = time.monotonic()
        try:
            result = self._tools.invoke(task, action.tool_name, dict(action.args))
            error = result.error
            value = None if error is not None else _clip(result.value)
        except Exception as failure:
            error = f"tool raised {type(failure).__name__}: {failure}"
            value = None
        return ToolCall(
            ordinal=ordinal,
            tool_name=action.tool_name,
            args=dict(action.args),
            result=value,
            error=error,
            elapsed_seconds=time.monotonic() - call_started,
            tokens=TokenUsage(
                prompt_tokens=max(0, action.prompt_tokens),
                completion_tokens=max(0, action.completion_tokens),
            ),
        )

    def run_suite(
        self, spec: AgentSpec, suite: DomainSuite, *, generation: int = 0
    ) -> EvaluationRun:
        """Run every task in the suite. Trajectory ids are unique per generation."""
        records = tuple(
            self.run_task(
                spec,
                task,
                trajectory_id=f"g{generation}::{spec.spec_id}::{task.task_id}",
            )
            for task in suite.tasks
        )
        return EvaluationRun(spec_id=spec.spec_id, task_set_id=suite.domain, records=records)

    def as_task_runner(self):
        """Expose this runner as the :class:`~agent_engineer.ports.TaskRunner` callable
        the measurement harness drives directly: ``(spec, task, attempt) -> Trajectory``.
        """

        def run(spec: AgentSpec, task: TaskSpec, attempt: int) -> Trajectory:
            return self.run_task(
                spec, task, trajectory_id=f"{spec.spec_id}::{task.task_id}::attempt{attempt}"
            ).trajectory

        return run
