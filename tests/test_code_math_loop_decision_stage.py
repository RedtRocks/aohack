"""Proves the loop's keep-or-revert decision, not just its wiring.

``test_code_math_loop_end_to_end.py`` crosses the integration gate with a
backend whose competence rises monotonically with generation, so every
mutation there measures a positive delta and every one is accepted. That is a
legitimate wiring test, but on its own it demonstrates nothing about the
decision the loop actually makes: a system that accepts everything is not
evaluating anything. This module exercises the two cases that matter:

* a mutation that makes the measured metric worse must be rejected, and the
  spec actually in force afterwards must be the parent, not merely a boolean
  saying so;
* a mutation whose measured delta is smaller than the run-to-run noise in the
  metric (a real reliability variance, measured over ``repeats=3`` through the
  frozen :class:`~agent_engineer.evaluation.Evaluator` harness, not asserted by
  fiat) must not be treated as an improvement. The engine's default
  :class:`~agent_engineer.stages.select.MinimumDeltaPolicy` has a ``min_delta``
  of ``1e-9`` -- effectively "any positive number" -- so this also demonstrates
  that the default policy alone does *not* protect against noise; only a
  caller who measures reliability and configures ``min_delta`` from it gets
  that protection. That gap is real and is called out in the limitations
  write-up, not papered over here.

Every backend below is scripted and says so. Nothing here calls a real model.
"""

from __future__ import annotations

import re

from agent_engineer.domains.code_math import DOMAIN, EVALUATOR_ID, get_evaluator, get_suite
from agent_engineer.evaluation import Evaluator
from agent_engineer.loop import run_loop
from agent_engineer.ports import AgentAction, ToolResult, ToolSchema
from agent_engineer.schemas import AgentSpec
from agent_engineer.stages.evaluate import TrajectoryRunner
from agent_engineer.stages.select import MinimumDeltaPolicy

_GEN_SUFFIX = re.compile(r"-g(\d+)$")
GOAL = "Solve each task exactly, in the exact output format the prompt asks for."


class _NoTools:
    """code_math needs no tools: every task is answered directly."""

    def schemas(self) -> tuple[ToolSchema, ...]:
        return ()

    def invoke(self, task, tool_name: str, args: dict) -> ToolResult:  # pragma: no cover
        return ToolResult(error=f"code_math exposes no tools; got {tool_name!r}")


def _tier(spec_id: str) -> int:
    """Generation number the loop itself burned into the spec id, or 0 for the root."""
    match = _GEN_SUFFIX.search(spec_id)
    return int(match.group(1)) if match else 0


class _TieredBackend:
    """Solves exactly ``solved_by_tier[tier]`` tasks correctly, deterministically,
    where tier is read off the spec id (see :func:`_tier`). Scripted, not a model:
    it exists to put a chosen, known number of tasks right or wrong per
    generation so the resulting lineage is exact and checkable by hand."""

    def __init__(self, task_order: tuple[str, ...], solved_by_tier: dict[int, int]) -> None:
        self._order = task_order
        self._solved_by_tier = solved_by_tier

    def next_action(self, spec, task, tools, history) -> AgentAction:
        tier = _tier(spec.spec_id)
        solved = self._solved_by_tier.get(tier, self._solved_by_tier[max(self._solved_by_tier)])
        solved_ids = set(self._order[:solved])
        answer = task.expected if task.task_id in solved_ids else None
        return AgentAction(final_answer=answer, prompt_tokens=5, completion_tokens=5)


class _SingleFlakyProbeBackend:
    """Scripted, used only to measure a real reliability variance through the
    frozen Evaluator harness (``repeats=3``). Exactly one task flips from
    correct to wrong depending on which attempt is asked; every other task is
    answered correctly on every attempt, so all of the measured variance comes
    from that one task, and the number is checkable by hand: the population
    variance of one flip among three attempts is 2/9 regardless of which
    attempt flips."""

    def __init__(self, flaky_task_id: str) -> None:
        self._flaky_task_id = flaky_task_id
        self._seen = 0

    def next_action(self, spec, task, tools, history) -> AgentAction:
        if task.task_id == self._flaky_task_id:
            correct = self._seen == 0
            self._seen += 1
        else:
            correct = True
        answer = task.expected if correct else None
        return AgentAction(final_answer=answer, prompt_tokens=5, completion_tokens=5)


def _run(backend, suite, *, max_generations: int, selection_policy=None, spec_id: str):
    return run_loop(
        spec_id=spec_id,
        goal=GOAL,
        tool_runtime=_NoTools(),
        backend=backend,
        evaluator=get_evaluator(),
        evaluator_id=EVALUATOR_ID,
        task_suite=suite,
        max_generations=max_generations,
        metric="mean_score",
        selection_policy=selection_policy,
    )


def test_a_mutation_that_makes_things_worse_is_rejected_and_the_parent_stays_in_force():
    suite = get_suite()
    task_order = tuple(task.task_id for task in suite.tasks)
    # tier 0 (root) solves 10/16; tier 1 (the only mutation tried) solves only 3/16
    backend = _TieredBackend(task_order, {0: 10, 1: 3})

    report = _run(backend, suite, max_generations=1, spec_id="worse-mutation-agent")

    assert len(report.generations) == 1
    record = report.generations[0]
    assert record.verdict.delta < 0.0
    assert record.verdict.before == 10 / 16
    assert record.verdict.after == 3 / 16
    assert not record.accepted
    assert record.verdict.decision.value == "reverted"

    # the load-bearing assertion: the spec actually in force afterwards is the
    # parent, not merely a "reverted" flag next to a child nobody is running
    assert report.final_spec.spec_id == report.root_spec.spec_id
    assert report.final_spec.spec_id != record.spec_after.spec_id
    assert report.final_spec.system_prompt == report.root_spec.system_prompt


def test_a_delta_smaller_than_measured_reliability_variance_is_not_an_improvement():
    suite = get_suite()
    task_order = tuple(task.task_id for task in suite.tasks)
    evaluator = get_evaluator()

    # Step 1: measure a REAL reliability variance through the frozen harness,
    # repeats=3, on a backend whose only source of variance is one task that
    # flips pass/fail depending on the attempt. This is not asserted by fiat --
    # it comes out of agent_engineer.evaluation.Evaluator itself.
    probe_spec = AgentSpec(spec_id="reliability-probe", system_prompt="Answer the task.")
    probe_backend = _SingleFlakyProbeBackend(task_order[0])
    probe_runner = TrajectoryRunner(backend=probe_backend, tool_runtime=_NoTools(), evaluator=evaluator)
    harness = Evaluator([suite], evaluators={DOMAIN: evaluator}, repeats=3)
    probe_report = harness.run_iteration(0, probe_spec, probe_runner.as_task_runner())
    reliability = probe_report.domain(DOMAIN).reliability
    assert reliability.is_defined
    # one flip among three attempts always has population variance 2/9, spread
    # over 16 tasks (only one of which varies at all)
    expected_variance = (2 / 9) / len(suite.tasks)
    assert abs(reliability.value - expected_variance) < 1e-9
    noise_std = reliability.value**0.5

    # Step 2: a deterministic, non-flaky mutation that is genuinely better --
    # one more task solved out of 16 -- but by less than the noise floor above.
    backend_kwargs = dict(
        backend=_TieredBackend(task_order, {0: 10, 1: 11}),
        suite=suite,
        max_generations=1,
    )
    true_delta = 1 / len(suite.tasks)
    assert true_delta < noise_std, "the scenario requires the true delta to sit inside the noise"

    # The engine's default policy (min_delta=1e-9) has no notion of noise: it
    # accepts any positive delta, including one smaller than the measured
    # variance. This is the real, current behaviour -- not a hypothetical.
    naive_report = _run(**backend_kwargs, selection_policy=None, spec_id="naive-agent")
    assert naive_report.generations[0].accepted
    assert abs(naive_report.generations[0].verdict.delta - true_delta) < 1e-9

    # Only when the caller measures reliability and configures min_delta from
    # it does the same true, positive delta get correctly withheld as noise.
    noise_aware_report = _run(
        **backend_kwargs,
        selection_policy=MinimumDeltaPolicy(min_delta=noise_std),
        spec_id="noise-aware-agent",
    )
    assert not noise_aware_report.generations[0].accepted
    assert noise_aware_report.final_spec.spec_id == noise_aware_report.root_spec.spec_id


def test_a_mixed_lineage_has_at_least_one_accept_and_one_revert():
    """A sanity check on the mixed run captured as the demo artifact
    (see experiments/generate_scripted_decision_lineage.py): not a staged
    all-accept run, and not a staged all-reject run either."""
    suite = get_suite()
    task_order = tuple(task.task_id for task in suite.tasks)
    probe_backend = _SingleFlakyProbeBackend(task_order[0])
    evaluator = get_evaluator()
    probe_runner = TrajectoryRunner(backend=probe_backend, tool_runtime=_NoTools(), evaluator=evaluator)
    harness = Evaluator([suite], evaluators={DOMAIN: evaluator}, repeats=3)
    noise_std = (
        harness.run_iteration(
            0, AgentSpec(spec_id="probe", system_prompt="Answer the task."), probe_runner.as_task_runner()
        )
        .domain(DOMAIN)
        .reliability.value
        ** 0.5
    )

    backend = _TieredBackend(task_order, {0: 10, 1: 3, 2: 13, 3: 14})
    report = _run(
        backend,
        suite,
        max_generations=3,
        selection_policy=MinimumDeltaPolicy(min_delta=noise_std),
        spec_id="mixed-lineage-agent",
    )

    decisions = [record.verdict.decision.value for record in report.generations]
    assert "accepted" in decisions
    assert "reverted" in decisions
    assert any(record.verdict.delta < 0 for record in report.generations)
    for record in report.generations:
        assert isinstance(record.verdict.before, float)
        assert record.verdict.delta == record.verdict.after - record.verdict.before
