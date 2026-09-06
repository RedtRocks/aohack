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
  fiat) must not be treated as an improvement.

That second case used to be a real gap: :func:`agent_engineer.loop.run_loop`
used to default to :class:`~agent_engineer.stages.select.MinimumDeltaPolicy`'s
own default (``min_delta=1e-9``), which has no notion of noise and accepts any
positive delta whatsoever. ``run_loop`` now measures a real noise floor for
the root spec automatically, before running any generation, and uses it as the
default keep threshold -- so a caller does not have to know to configure
anything to get this protection; see the module docstring on
``agent_engineer/loop.py``. A caller who wants the old, permissive behaviour
has to opt into it explicitly by passing their own ``selection_policy`` --
which also means the automatic measurement (and its extra calls) never runs
for them, so opting out is free.

Every backend below is scripted and says so. Nothing here calls a real model.
"""

from __future__ import annotations

import re

from agent_engineer.domains.code_math import EVALUATOR_ID, get_evaluator, get_suite
from agent_engineer.loop import run_loop
from agent_engineer.ports import AgentAction, ToolResult, ToolSchema
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
    generation so the resulting lineage is exact and checkable by hand. Fully
    deterministic -- repeated attempts of the same spec always agree -- so it
    measures zero reliability variance through the harness."""

    def __init__(self, task_order: tuple[str, ...], solved_by_tier: dict[int, int]) -> None:
        self._order = task_order
        self._solved_by_tier = solved_by_tier

    def next_action(self, spec, task, tools, history) -> AgentAction:
        tier = _tier(spec.spec_id)
        solved = self._solved_by_tier.get(tier, self._solved_by_tier[max(self._solved_by_tier)])
        solved_ids = set(self._order[:solved])
        answer = task.expected if task.task_id in solved_ids else None
        return AgentAction(final_answer=answer, prompt_tokens=5, completion_tokens=5)


class _TieredWithNoiseBackend:
    """Like :class:`_TieredBackend`, plus exactly one task with genuine,
    hand-scripted variance.

    That one task's correctness follows ``noisy_pattern`` for its first calls
    (enough entries to give the frozen Evaluator harness's ``repeats=3``
    measurement real, non-zero variance to find) and is constant (correct)
    after the pattern runs out. Since ``run_loop``'s own default noise-floor
    measurement is the *first* thing that touches this backend -- three calls
    to every task, before any generation runs -- the noisy task's pattern is
    exactly consumed by that measurement, and every generation-time call
    afterwards lands past the end of the pattern and is constant. The tiered
    structural signal is therefore the only thing that varies from one
    generation's comparison to the next; the noise floor it gets compared
    against is real, not asserted.
    """

    def __init__(
        self,
        task_order: tuple[str, ...],
        solved_by_tier: dict[int, int],
        *,
        noisy_task_id: str,
        noisy_pattern: tuple[bool, ...] = (True, False, True),
    ) -> None:
        self._other_tasks = tuple(t for t in task_order if t != noisy_task_id)
        self._solved_by_tier = solved_by_tier
        self._noisy_task_id = noisy_task_id
        self._noisy_pattern = noisy_pattern
        self._noisy_calls = 0

    def next_action(self, spec, task, tools, history) -> AgentAction:
        if task.task_id == self._noisy_task_id:
            index = self._noisy_calls
            self._noisy_calls += 1
            correct = self._noisy_pattern[index] if index < len(self._noisy_pattern) else True
        else:
            tier = _tier(spec.spec_id)
            solved = self._solved_by_tier.get(tier, self._solved_by_tier[max(self._solved_by_tier)])
            correct = task.task_id in set(self._other_tasks[:solved])
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


def test_the_default_reverts_a_delta_smaller_than_the_measured_noise_floor():
    """The fix: run_loop's own default now measures a real reliability variance
    for the root spec and reverts a sub-noise delta WITHOUT the caller having
    to configure anything."""
    suite = get_suite()
    task_order = tuple(task.task_id for task in suite.tasks)
    # 9/15 non-noisy tasks solved at tier 0, 10/15 at tier 1 -- plus the noisy
    # task, constant-correct from generation 0 onward (see class docstring) --
    # gives a true delta of exactly 1/16, checkable by hand.
    backend = _TieredWithNoiseBackend(
        task_order, {0: 9, 1: 10}, noisy_task_id=task_order[0]
    )

    report = _run(backend, suite, max_generations=1, spec_id="noise-floor-agent")

    # the noise floor was measured, not asserted: one flip among three attempts
    # always has population variance 2/9, spread over 16 tasks
    assert report.noise_floor is not None
    expected_noise_floor = ((2 / 9) / len(suite.tasks)) ** 0.5
    assert abs(report.noise_floor - expected_noise_floor) < 1e-9

    record = report.generations[0]
    true_delta = 1 / len(suite.tasks)
    assert abs(record.verdict.delta - true_delta) < 1e-9
    assert true_delta < report.noise_floor, "the scenario requires the true delta to sit inside the noise"

    # the load-bearing assertion: reverted BY DEFAULT, no selection_policy passed
    assert not record.accepted
    assert report.final_spec.spec_id == report.root_spec.spec_id


def test_explicit_opt_out_keeps_the_old_permissive_behaviour_and_skips_measurement():
    """A caller who wants the old, fixed-epsilon behaviour still gets it --
    deliberately, by passing their own selection_policy -- and that choice
    also means no noise-floor measurement (and no extra calls) ever runs."""
    suite = get_suite()
    task_order = tuple(task.task_id for task in suite.tasks)
    backend = _TieredBackend(task_order, {0: 10, 1: 11})  # true delta 1/16, no noise involved

    report = _run(
        backend,
        suite,
        max_generations=1,
        selection_policy=MinimumDeltaPolicy(min_delta=1e-9),
        spec_id="opt-out-agent",
    )

    assert report.noise_floor is None  # the measurement never ran
    assert report.generations[0].accepted
    assert abs(report.generations[0].verdict.delta - 1 / len(suite.tasks)) < 1e-9


def test_a_mixed_lineage_has_at_least_one_accept_and_one_revert():
    """A sanity check on the mixed run captured as the demo artifact
    (see experiments/generate_scripted_decision_lineage.py): not a staged
    all-accept run, and not a staged all-reject run either, and reverts the
    sub-noise generation BY DEFAULT."""
    suite = get_suite()
    task_order = tuple(task.task_id for task in suite.tasks)
    # translates the original 16-task tiers {0:10,1:3,2:13,3:14} into the
    # 15-task (non-noisy) space; the noisy task adds a constant +1 from
    # generation 0 onward, so the totals land back on the same numbers
    backend = _TieredWithNoiseBackend(
        task_order, {0: 9, 1: 2, 2: 12, 3: 13}, noisy_task_id=task_order[0]
    )
    report = _run(backend, suite, max_generations=3, spec_id="mixed-lineage-agent")

    decisions = [record.verdict.decision.value for record in report.generations]
    assert "accepted" in decisions
    assert "reverted" in decisions
    assert any(record.verdict.delta < 0 for record in report.generations)
    for record in report.generations:
        assert isinstance(record.verdict.before, float)
        assert record.verdict.delta == record.verdict.after - record.verdict.before
    # gen 3 is the sub-noise case, reverted by default with no configuration
    assert report.generations[-1].verdict.decision.value == "reverted"
    assert abs(report.generations[-1].verdict.delta) < report.noise_floor
