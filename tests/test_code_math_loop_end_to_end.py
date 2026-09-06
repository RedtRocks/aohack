"""Integration-gate proof: the real five-stage loop, run against the real code_math domain.

Unlike :mod:`tests.test_loop_end_to_end`, which proves the engine's control
flow with a scripted stand-in domain, this module wires :func:`agent_engineer.loop.run_loop`
to the actual ``agent_engineer.domains.code_math`` suite and evaluator -- the
same objects the domain package hands the measurement harness. code_math is
the domain the gate is crossed on first because its evaluator settles every
answer by execution or exact comparison, never by judgement, so a wiring bug
between the engine and the evaluator shows up as a wrong score rather than
hiding behind a lenient judge.

The backend below is a scripted stand-in for a model, exactly as
``_ImprovingBackend`` is in the other end-to-end test: it does not implement an
agent, it simulates one whose competence rises with each accepted generation,
so the loop's own mutate/select machinery is what is under test, not a model.
It reads ``task.expected`` to decide whether *this run* is one it gets right --
the same shortcut ``_ImprovingBackend`` takes with ``task.metadata["key"]`` --
which is legitimate for a test double proving engine plumbing, and is
distinct from the domain's own baseline tests (``test_code_math_domain.py``),
which never look at ground truth because they are proving the domain gives a
real agent a usable, non-zero signal.
"""

from __future__ import annotations

import re

from agent_engineer.domains.code_math import EVALUATOR_ID, get_evaluator, get_suite
from agent_engineer.loop import run_loop
from agent_engineer.ports import AgentAction, ToolResult, ToolSchema

_GEN_SUFFIX = re.compile(r"-g(\d+)$")


class _NoTools:
    """code_math needs no tools: every task is answered directly."""

    def schemas(self) -> tuple[ToolSchema, ...]:
        return ()

    def invoke(self, task, tool_name: str, args: dict) -> ToolResult:  # pragma: no cover
        return ToolResult(error=f"code_math exposes no tools; got {tool_name!r}")


class _TieredCodeMathBackend:
    """Answers more tasks correctly as the lineage advances.

    Tier is read off the generation number the loop itself burned into
    ``child_spec_id`` (``f"{spec_id}-g{generation}"`` in :func:`run_loop`), not
    off the content of any particular mutation -- so this stands in for
    "each accepted generation of tuning made the agent a little more capable"
    without hardcoding which kind of mutation the ladder happens to propose.
    """

    def __init__(self, task_order: tuple[str, ...], *, base: int = 2, step: int = 3) -> None:
        self._order = task_order
        self._base = base
        self._step = step

    def _tier(self, spec_id: str) -> int:
        match = _GEN_SUFFIX.search(spec_id)
        return int(match.group(1)) if match else 0

    def next_action(self, spec, task, tools, history) -> AgentAction:
        solved = min(len(self._order), self._base + self._step * self._tier(spec.spec_id))
        solved_ids = set(self._order[:solved])
        answer = task.expected if task.task_id in solved_ids else None
        return AgentAction(final_answer=answer, prompt_tokens=30, completion_tokens=20)


def test_five_stage_loop_runs_end_to_end_on_code_math():
    """The gate: a real lineage of >=3 mutations against the real code_math domain,
    each carrying its measured delta and the before number it improved on."""
    suite = get_suite()
    task_order = tuple(task.task_id for task in suite.tasks)
    assert len(task_order) >= 11, "the tiered backend assumes the full 16-task suite"

    report = run_loop(
        spec_id="code-math-agent",
        goal="Solve each task exactly, in the exact output format the prompt asks for.",
        tool_runtime=_NoTools(),
        backend=_TieredCodeMathBackend(task_order),
        evaluator=get_evaluator(),
        evaluator_id=EVALUATOR_ID,
        task_suite=suite,
        max_generations=4,
        metric="mean_score",
    )

    # the gate's own shape: at least three mutations, each accepted-or-rejected,
    # each carrying a measured delta and the before number it improved on
    assert len(report.generations) >= 3
    for record in report.generations:
        assert record.verdict.decision.value in {"accepted", "reverted"}
        assert isinstance(record.verdict.before, float)
        assert record.verdict.delta == record.verdict.after - record.verdict.before
        # every mutation is traceable to the diagnosis and spec that motivated it
        assert record.mutation.motivating_diagnosis.spec_id == record.spec_before.spec_id
        assert record.mutation.before != record.mutation.after

    # the tiered backend genuinely improves the measured metric each generation
    for record in report.generations:
        assert record.verdict.after > record.verdict.before
        assert record.accepted
    assert report.final_spec.spec_id != report.root_spec.spec_id

    # cost and speed are read only off Trajectory.tokens / elapsed_seconds -- confirm
    # every recorded trajectory populated both, on every path, not just the happy one
    for run in (report.generations[0].evaluation_before, report.generations[-1].evaluation_after):
        for trajectory in run.trajectories:
            assert trajectory.tokens.total_tokens > 0
            assert trajectory.elapsed_seconds >= 0.0

    print("\n".join(report.summary_lines()))
