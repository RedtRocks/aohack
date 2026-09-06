"""End-to-end smoke test: the loop runs on one scripted domain and produces a lineage.

This is not a domain implementation -- it is a minimal stand-in for whatever a
real ``DomainSuite``/``TaskEvaluator`` pair from ``agent_engineer.evaluation``
looks like, built only to prove the engine's own control flow (stages 1-5,
wired by :func:`agent_engineer.loop.run_loop`) end to end without a live model
or a real domain package. Nothing below is engine code; it lives in tests
because the engine must never import anything like it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_engineer.loop import run_loop
from agent_engineer.ports import AgentAction, ToolResult, ToolSchema
from agent_engineer.schemas import EvaluatorVerdict


@dataclass(frozen=True)
class _Task:
    task_id: str
    prompt: str
    answer: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _Suite:
    domain: str
    tasks: tuple[_Task, ...]


class _LookupTool:
    """A single tool: looks a value up in a fixed table. Deterministic, no network."""

    _TABLE = {"a": "1", "b": "2", "c": "3"}

    def schemas(self) -> tuple[ToolSchema, ...]:
        return (
            ToolSchema(
                name="lookup",
                description="Look up a key in the table.",
                parameters={
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                },
            ),
        )

    def invoke(self, task, tool_name: str, args: dict) -> ToolResult:
        if tool_name != "lookup":
            return ToolResult(error=f"no such tool {tool_name!r}")
        key = args.get("key")
        if key not in self._TABLE:
            return ToolResult(error=f"unknown key {key!r}")
        return ToolResult(value=self._TABLE[key])


class _ImprovingBackend:
    """A scripted backend whose competence increases with each system-prompt correction.

    Behaviour is driven purely by how many correction blocks the loop has
    appended to the system prompt so far (a structural signal, counted, not
    read for content), so this stands in for "the mutations are actually
    helping" without hardcoding anything about a domain.
    """

    def next_action(self, spec, task, tools, history):
        corrections = spec.system_prompt.count("Correction, from observed failures")
        if not history:
            if corrections == 0:
                # Generation 0: call the tool with the wrong key entirely.
                return AgentAction(tool_name="lookup", args={"key": "z"}, prompt_tokens=10, completion_tokens=5)
            # Corrected: call the tool with the right key.
            return AgentAction(
                tool_name="lookup", args={"key": task.metadata["key"]}, prompt_tokens=10, completion_tokens=5
            )
        last = history[-1]
        if last.succeeded:
            return AgentAction(final_answer=last.result, prompt_tokens=5, completion_tokens=5)
        return AgentAction(final_answer="unknown", prompt_tokens=5, completion_tokens=5)


def _evaluator(task: _Task, trajectory) -> EvaluatorVerdict:
    passed = trajectory.final_answer == task.answer
    return EvaluatorVerdict(
        evaluator_id="exact-match",
        passed=passed,
        score=1.0 if passed else 0.0,
        rationale="exact match" if passed else f"expected {task.answer!r}, got {trajectory.final_answer!r}",
    )


def _suite() -> _Suite:
    tasks = tuple(
        _Task(task_id=f"t{i}", prompt=f"look up {key}", answer=value, metadata={"key": key})
        for i, (key, value) in enumerate(_LookupTool._TABLE.items())
    )
    return _Suite(domain="lookup-suite", tasks=tasks)


def test_loop_runs_end_to_end_and_produces_a_lineage():
    report = run_loop(
        spec_id="agent-0",
        goal="Answer each question using the lookup tool.",
        tool_runtime=_LookupTool(),
        backend=_ImprovingBackend(),
        evaluator=_evaluator,
        evaluator_id="exact-match",
        task_suite=_suite(),
        max_generations=3,
    )

    assert len(report.generations) >= 1
    for record in report.generations:
        # every generation carries its measured delta AND its before number
        assert isinstance(record.verdict.before, float)
        assert isinstance(record.verdict.delta, float)
        assert record.mutation.motivating_diagnosis.spec_id == record.spec_before.spec_id

    # the scripted backend gets it right as soon as one correction lands
    assert report.generations[0].verdict.after > report.generations[0].verdict.before
    assert report.generations[0].accepted
    assert report.final_spec.spec_id != report.root_spec.spec_id


def test_lineage_of_two_mutations_then_an_honest_stall():
    """The integration-gate shape: every ladder move tried gets accepted-or-rejected
    with before+delta, and the loop stops the instant the ladder for the
    diagnosed cause runs out, rather than reaching for an unrelated edit to
    keep generating.

    A backend that always misuses the tool the same way exhausts the
    tool-misuse ladder -- prompt guidance, then a loop-shape escalation -- in
    exactly two generations. There is deliberately no third generation: the
    ladder does not fall back to a step-budget raise or a memory change for a
    cause neither addresses, so the run stalls honestly instead of padding the
    lineage with a mutation whose rationale would not actually be true.
    """

    class _NeverLearningBackend:
        """Always calls the wrong key; never improves, so nothing here is ever accepted."""

        def next_action(self, spec, task, tools, history):
            if not history:
                return AgentAction(tool_name="lookup", args={"key": "z"}, prompt_tokens=1, completion_tokens=1)
            return AgentAction(final_answer="unknown", prompt_tokens=1, completion_tokens=1)

    report = run_loop(
        spec_id="agent-stall",
        goal="Answer each question using the lookup tool.",
        tool_runtime=_LookupTool(),
        backend=_NeverLearningBackend(),
        evaluator=_evaluator,
        evaluator_id="exact-match",
        task_suite=_suite(),
        max_generations=5,
    )

    assert len(report.generations) == 2  # the tool-misuse ladder has exactly two honest moves
    for record in report.generations:
        assert record.verdict.decision.value in {"accepted", "reverted"}
        assert record.verdict.before == 0.0
        assert record.verdict.delta == record.verdict.after - record.verdict.before
        assert not record.accepted  # nothing improves a backend that never learns
    for line in report.summary_lines():
        assert "->" in line
