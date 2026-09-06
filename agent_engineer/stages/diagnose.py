"""Stage 3: attribute each *failing* trajectory to one :class:`FailureCause`.

Passing trajectories are never diagnosed. Unevaluated trajectories are never
diagnosed either -- absence of a verdict is missing measurement, not failure.
That is what :attr:`Trajectory.failed` already encodes, and this stage takes it
at its word rather than re-deriving success from the answer text.

Every rule below reads only structural signals that exist in any domain: how
the run stopped, whether tools errored, whether calls repeated verbatim,
whether the evaluator granted partial credit. No rule inspects the *content* of
a prompt, an answer, or a tool result for domain vocabulary -- doing so would
make the diagnoser a domain module wearing a general-purpose name.

The rules are ordered and the first match wins, so each failure gets exactly
one cause, as :class:`Diagnosis` requires. The last rule always matches, so the
closed enum stays closed: nothing falls through to a bucket.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, runtime_checkable

from agent_engineer.ports import TextGenerator
from agent_engineer.schemas import (
    AgentSpec,
    Diagnosis,
    FailureAttribution,
    FailureCause,
    MemoryKind,
    Trajectory,
)
from agent_engineer.stages.evaluate import EvaluationRun, RunRecord, StopReason

PARTIAL_CREDIT_THRESHOLD = 0.5
"""Score at or above which a failed run counts as substantively right but badly presented."""


@runtime_checkable
class Diagnoser(Protocol):
    """Turns the failures of one evaluation run into a :class:`Diagnosis`."""

    def diagnose(self, spec: AgentSpec, run: EvaluationRun, *, diagnosis_id: str) -> Diagnosis: ...


@dataclass(frozen=True)
class _Finding:
    cause: FailureCause
    evidence: str
    confidence: float = 1.0
    step_ordinal: int | None = None


Rule = Callable[[AgentSpec, RunRecord], "_Finding | None"]


def _errored(trajectory: Trajectory) -> tuple[int, ...]:
    return tuple(call.ordinal for call in trajectory.tool_calls if not call.succeeded)


def _first_repeat(trajectory: Trajectory) -> int | None:
    """Ordinal of the first call that repeats an earlier call verbatim, if any."""
    seen: set[tuple[str, str]] = set()
    for call in trajectory.tool_calls:
        key = (call.tool_name, json.dumps(call.args, sort_keys=True, default=str))
        if key in seen:
            return call.ordinal
        seen.add(key)
    return None


def _rule_budget(spec: AgentSpec, record: RunRecord) -> _Finding | None:
    if not record.budget_exhausted:
        return None
    return _Finding(
        cause=FailureCause.STOPPING_CONDITION_HIT,
        evidence=(
            f"the run was cut off by {record.stop_reason.value} after "
            f"{record.trajectory.step_count} tool calls and "
            f"{record.trajectory.tokens.total_tokens} tokens, with no final answer"
            if record.trajectory.final_answer is None
            else f"the run was cut off by {record.stop_reason.value} after "
            f"{record.trajectory.step_count} tool calls"
        ),
        confidence=1.0,
    )


def _rule_backend_error(spec: AgentSpec, record: RunRecord) -> _Finding | None:
    if record.stop_reason is not StopReason.BACKEND_ERROR:
        return None
    last = record.trajectory.tool_calls[-1] if record.trajectory.tool_calls else None
    return _Finding(
        cause=FailureCause.FAULTY_REASONING,
        evidence=f"the model failed to produce a usable next action: {last.error if last else 'unknown'}",
        confidence=0.6,
        step_ordinal=last.ordinal if last else None,
    )


def _rule_unknown_tool(spec: AgentSpec, record: RunRecord) -> _Finding | None:
    exposed = set(spec.tools)
    for call in record.trajectory.tool_calls:
        if not call.succeeded and call.tool_name not in exposed:
            return _Finding(
                cause=FailureCause.WRONG_TOOL_SELECTED,
                evidence=(
                    f"step {call.ordinal} called {call.tool_name!r}, which this spec does not "
                    f"expose; available: {', '.join(spec.tools) or 'none'}"
                ),
                step_ordinal=call.ordinal,
            )
    return None


def _rule_repeated_tool_errors(spec: AgentSpec, record: RunRecord) -> _Finding | None:
    failures = [call for call in record.trajectory.tool_calls if not call.succeeded]
    if len(failures) < 2:
        return None
    return _Finding(
        cause=FailureCause.TOOL_MISUSE,
        evidence=(
            f"{len(failures)} of {record.trajectory.step_count} tool calls errored "
            f"(steps {', '.join(str(call.ordinal) for call in failures)}); "
            f"the first said: {failures[0].error}"
        ),
        step_ordinal=failures[0].ordinal,
    )


def _rule_unrecovered_tool_error(spec: AgentSpec, record: RunRecord) -> _Finding | None:
    calls = record.trajectory.tool_calls
    if not calls or calls[-1].succeeded:
        return None
    return _Finding(
        cause=FailureCause.TOOL_MISUSE,
        evidence=(
            f"the last tool call (step {calls[-1].ordinal}, {calls[-1].tool_name!r}) errored and "
            f"the run ended without recovering from it: {calls[-1].error}"
        ),
        confidence=0.9,
        step_ordinal=calls[-1].ordinal,
    )


def _rule_no_tools_used(spec: AgentSpec, record: RunRecord) -> _Finding | None:
    if record.trajectory.tool_calls or not spec.tools:
        return None
    if record.trajectory.final_answer is None:
        return _Finding(
            cause=FailureCause.PREMATURE_STOP,
            evidence="the run ended with no tool calls and no final answer",
        )
    return _Finding(
        cause=FailureCause.WRONG_TOOL_SELECTED,
        evidence=(
            f"answered directly without calling any of the {len(spec.tools)} available tools "
            f"({', '.join(spec.tools)})"
        ),
    )


def _rule_partial_credit(spec: AgentSpec, record: RunRecord) -> _Finding | None:
    verdict = record.trajectory.verdict
    if verdict is None or verdict.score < PARTIAL_CREDIT_THRESHOLD:
        return None
    if record.trajectory.final_answer is None:
        return None
    return _Finding(
        cause=FailureCause.OUTPUT_FORMAT_VIOLATION,
        evidence=(
            f"the evaluator scored {verdict.score:.2f} but still failed the run, so the work was "
            f"largely right and the answer's shape was not: {verdict.rationale or 'no rationale given'}"
        ),
        confidence=0.8,
    )


def _rule_repeated_call(spec: AgentSpec, record: RunRecord) -> _Finding | None:
    ordinal = _first_repeat(record.trajectory)
    if ordinal is None:
        return None
    call = record.trajectory.tool_calls[ordinal]
    return _Finding(
        cause=FailureCause.CONTEXT_LOSS,
        evidence=(
            f"step {ordinal} repeated an earlier call to {call.tool_name!r} with identical "
            "arguments, so the earlier result was not carried forward"
        ),
        confidence=0.8,
        step_ordinal=ordinal,
    )


def _rule_lossy_memory(spec: AgentSpec, record: RunRecord) -> _Finding | None:
    lossy = {MemoryKind.NONE, MemoryKind.ROLLING_SUMMARY, MemoryKind.SCRATCHPAD}
    if spec.memory.kind not in lossy or record.trajectory.step_count < 3:
        return None
    return _Finding(
        cause=FailureCause.CONTEXT_LOSS,
        evidence=(
            f"the run took {record.trajectory.step_count} steps under memory kind "
            f"{spec.memory.kind.value!r}, which does not retain earlier observations in full"
        ),
        confidence=0.5,
    )


def _rule_stopped_early(spec: AgentSpec, record: RunRecord) -> _Finding | None:
    if record.stop_reason is not StopReason.FINAL_ANSWER:
        return None
    budget = spec.stopping.max_steps
    if budget is None or record.trajectory.step_count > budget // 2:
        return None
    return _Finding(
        cause=FailureCause.PREMATURE_STOP,
        evidence=(
            f"answered after {record.trajectory.step_count} of an available {budget} steps, "
            "leaving most of the budget unused"
        ),
        confidence=0.6,
    )


def _rule_default(spec: AgentSpec, record: RunRecord) -> _Finding:
    verdict = record.trajectory.verdict
    score = verdict.score if verdict else 0.0
    rationale = (verdict.rationale if verdict else "") or "no rationale given"
    return _Finding(
        cause=FailureCause.FAULTY_REASONING,
        evidence=(
            f"the run completed {record.trajectory.step_count} tool calls without error and "
            f"answered, but the evaluator scored {score:.2f} and failed it: {rationale}"
        ),
        confidence=0.7,
    )


DEFAULT_RULES: tuple[Rule, ...] = (
    _rule_budget,
    _rule_backend_error,
    _rule_unknown_tool,
    _rule_repeated_tool_errors,
    _rule_unrecovered_tool_error,
    _rule_no_tools_used,
    _rule_partial_credit,
    _rule_repeated_call,
    _rule_lossy_memory,
    _rule_stopped_early,
)
"""Ordered, first-match-wins. ``_rule_default`` runs when none of these match."""


class HeuristicDiagnoser:
    """Rule-based attribution over structural trajectory signals. No model call."""

    def __init__(self, rules: Iterable[Rule] = DEFAULT_RULES) -> None:
        self._rules = tuple(rules)

    def attribute(self, spec: AgentSpec, record: RunRecord) -> FailureAttribution:
        """Attribute one failing run to exactly one cause."""
        finding = next(
            (found for rule in self._rules if (found := rule(spec, record)) is not None),
            None,
        ) or _rule_default(spec, record)
        return FailureAttribution(
            trajectory_id=record.trajectory.trajectory_id,
            cause=finding.cause,
            evidence=finding.evidence,
            confidence=finding.confidence,
            step_ordinal=finding.step_ordinal,
        )

    def diagnose(self, spec: AgentSpec, run: EvaluationRun, *, diagnosis_id: str) -> Diagnosis:
        failing = tuple(record for record in run.records if record.trajectory.failed)
        attributions = tuple(self.attribute(spec, record) for record in failing)
        return Diagnosis(
            diagnosis_id=diagnosis_id,
            spec_id=spec.spec_id,
            attributions=attributions,
            notes=(
                f"{len(failing)} of {len(run.records)} runs on task set {run.task_set_id!r} "
                f"failed (pass rate {run.pass_rate:.2f}, mean score {run.mean_score:.2f})"
            ),
        )


_CAUSE_BY_VALUE = {cause.value: cause for cause in FailureCause}


class ModelDiagnoser:
    """Model-assisted attribution, constrained to the same closed enum.

    The model is shown the trajectory record and must answer with one enum value
    and a line of evidence. Anything it returns that is not one of the seven
    causes is discarded and the heuristic attribution stands -- the closed enum
    is enforced here, not hoped for.
    """

    _SYSTEM = (
        "You attribute one failed run of a tool-using agent to exactly one cause. "
        "Reply with two lines and nothing else:\n"
        "cause: <one of the listed cause values, verbatim>\n"
        "evidence: <one sentence citing what happened in the run>"
    )

    def __init__(self, generator: TextGenerator, *, fallback: HeuristicDiagnoser | None = None) -> None:
        self._generator = generator
        self._fallback = fallback or HeuristicDiagnoser()

    @staticmethod
    def _render(spec: AgentSpec, record: RunRecord) -> str:
        trajectory = record.trajectory
        lines = [
            "Allowed cause values, with what each means:",
            *(f"- {cause.value}: {' '.join((cause.__doc__ or '').split())}" for cause in FailureCause),
            "",
            f"Agent tools: {', '.join(spec.tools) or 'none'}",
            f"Strategy: {spec.strategy.value}; memory: {spec.memory.kind.value}",
            f"Stop reason: {record.stop_reason.value}",
            "",
            "Tool calls:",
        ]
        for call in trajectory.tool_calls:
            outcome = f"ERROR {call.error}" if call.error else f"ok -> {call.result!r}"
            lines.append(f"  [{call.ordinal}] {call.tool_name}({call.args!r}) {outcome}")
        if not trajectory.tool_calls:
            lines.append("  (none)")
        verdict = trajectory.verdict
        lines += [
            "",
            f"Final answer: {trajectory.final_answer!r}",
            f"Evaluator: passed={verdict.passed if verdict else None} "
            f"score={verdict.score if verdict else None} "
            f"rationale={(verdict.rationale if verdict else '')!r}",
        ]
        return "\n".join(lines)

    def attribute(self, spec: AgentSpec, record: RunRecord) -> FailureAttribution:
        baseline = self._fallback.attribute(spec, record)
        try:
            reply = self._generator.complete(self._SYSTEM, self._render(spec, record))
        except Exception:
            return baseline
        cause: FailureCause | None = None
        evidence = ""
        for line in (reply or "").splitlines():
            key, _, value = line.partition(":")
            key, value = key.strip().lower(), value.strip()
            if key == "cause":
                cause = _CAUSE_BY_VALUE.get(value.strip("'\" "))
            elif key == "evidence" and value:
                evidence = value
        if cause is None:
            return baseline
        return FailureAttribution(
            trajectory_id=record.trajectory.trajectory_id,
            cause=cause,
            evidence=evidence or baseline.evidence,
            confidence=0.9,
            step_ordinal=baseline.step_ordinal,
        )

    def diagnose(self, spec: AgentSpec, run: EvaluationRun, *, diagnosis_id: str) -> Diagnosis:
        failing = tuple(record for record in run.records if record.trajectory.failed)
        return Diagnosis(
            diagnosis_id=diagnosis_id,
            spec_id=spec.spec_id,
            attributions=tuple(self.attribute(spec, record) for record in failing),
            notes=(
                f"{len(failing)} of {len(run.records)} runs on task set {run.task_set_id!r} "
                f"failed (pass rate {run.pass_rate:.2f}, mean score {run.mean_score:.2f})"
            ),
        )
