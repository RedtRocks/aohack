"""Tests for the evaluator harness, the four metrics, and the results tables.

The three non-negotiable rules each get their own section. They are the ones a
reviewer will look for, and the ones that quietly ruin a demo if they regress.
"""

from __future__ import annotations

import pytest

from agent_engineer.evaluation import (
    DomainSuite,
    Evaluator,
    IterationReport,
    Metric,
    TaskSpec,
    is_refusal,
    mean_of_defined,
    population_variance,
    undefined,
)
from agent_engineer.evaluation.report import (
    UNDEFINED_CELL,
    Improvement,
    format_accuracy_with_cost,
    format_metric,
    render_iteration_table,
    render_progress_table,
    render_report,
)
from agent_engineer.schemas import AgentSpec, EvaluatorVerdict, TokenUsage, Trajectory

SPEC = AgentSpec(spec_id="spec-1", system_prompt="do the thing")


def task(task_id: str, domain: str = "extraction") -> TaskSpec:
    return TaskSpec(task_id=task_id, domain=domain, prompt=f"solve {task_id}")


def suite(domain: str = "extraction", count: int = 2) -> DomainSuite:
    return DomainSuite(domain=domain, tasks=tuple(task(f"t{i}", domain) for i in range(count)))


def trajectory(
    task_id: str,
    *,
    passed: bool = True,
    answer: str | None = "42",
    tokens: int = 100,
    seconds: float = 1.0,
    graded: bool = True,
) -> Trajectory:
    verdict = (
        EvaluatorVerdict(evaluator_id="test", passed=passed, score=1.0 if passed else 0.0)
        if graded
        else None
    )
    return Trajectory(
        trajectory_id=f"traj-{task_id}",
        spec_id=SPEC.spec_id,
        task_id=task_id,
        tokens=TokenUsage(prompt_tokens=tokens, completion_tokens=0),
        elapsed_seconds=seconds,
        final_answer=answer,
        verdict=verdict,
    )


def runner_returning(**by_task_id: Trajectory):
    def runner(spec: AgentSpec, task_spec: TaskSpec, attempt: int) -> Trajectory:
        return by_task_id[task_spec.task_id]

    return runner


def always(**kwargs):
    def runner(spec: AgentSpec, task_spec: TaskSpec, attempt: int) -> Trajectory:
        return trajectory(task_spec.task_id, **kwargs)

    return runner


# --------------------------------------------------------------------------- #
# RULE 1: accuracy is over ALL tasks, and a refusal is a FAILURE
# --------------------------------------------------------------------------- #


def test_accuracy_denominator_counts_every_task_not_just_attempted_ones():
    """An agent that answers one task and declines the other scores 0.5, not 1.0."""

    def runner(spec, task_spec, attempt):
        if task_spec.task_id == "t0":
            return trajectory("t0", passed=True)
        return trajectory("t1", answer="I can't help with that.", passed=False)

    report = Evaluator([suite(count=2)]).run_iteration(0, SPEC, runner)
    accuracy = report.domain("extraction").accuracy

    assert accuracy.require() == pytest.approx(0.5)
    # The denominator is every task times every repeat -- not the ones it tried.
    assert accuracy.sample_size == 2 * 3


def test_refusal_counts_as_failure_and_cannot_be_scored_as_a_pass():
    """Declining every task scores 0.0. It does not score, and it is not excluded."""
    report = Evaluator([suite(count=2)]).run_iteration(
        0, SPEC, always(answer="I'm unable to do that.", passed=False)
    )
    domain = report.domain("extraction")

    assert domain.accuracy.require() == 0.0
    assert domain.accuracy.sample_size == 6
    assert domain.refusal_count == 6


def test_refusal_overrides_a_verdict_claiming_it_passed():
    """A declined run is a failure even if something graded it as a pass."""
    report = Evaluator([suite(count=1)]).run_iteration(
        0, SPEC, always(answer="I cannot assist with this.", passed=True)
    )
    assert report.domain("extraction").accuracy.require() == 0.0


def test_declining_the_hard_task_never_beats_attempting_and_missing():
    """The whole point of rule 1: ducking a task must not pay."""
    attempts_and_misses = Evaluator([suite(count=2)]).run_iteration(
        0, SPEC, always(answer="wrong answer", passed=False)
    )
    declines = Evaluator([suite(count=2)]).run_iteration(
        0, SPEC, always(answer="I can't help with that.", passed=False)
    )
    assert declines.domain("extraction").accuracy.require() <= attempts_and_misses.domain(
        "extraction"
    ).accuracy.require()


def test_empty_and_missing_answers_are_refusals():
    assert is_refusal(trajectory("t", answer=None))
    assert is_refusal(trajectory("t", answer="   "))
    assert is_refusal(trajectory("t", answer="I cannot complete this task."))
    assert not is_refusal(trajectory("t", answer="42"))


def test_a_runner_that_raises_is_a_failure_still_in_the_denominator():
    """A crashed run must not vanish from the denominator and inflate accuracy."""

    def runner(spec, task_spec, attempt):
        if task_spec.task_id == "t1":
            raise RuntimeError("boom")
        return trajectory("t0", passed=True)

    domain = Evaluator([suite(count=2)]).run_iteration(0, SPEC, runner).domain("extraction")

    assert domain.accuracy.require() == pytest.approx(0.5)
    assert domain.accuracy.sample_size == 6
    assert all(outcome.errored for outcome in domain.results[1].outcomes)


def test_an_ungraded_run_is_not_a_pass():
    """Nobody confirmed it, so it does not count as a success."""
    report = Evaluator([suite(count=1)]).run_iteration(0, SPEC, always(graded=False))
    assert report.domain("extraction").accuracy.require() == 0.0


def test_domain_evaluator_is_used_when_the_trajectory_carries_no_verdict():
    def judge(task_spec: TaskSpec, traj: Trajectory) -> EvaluatorVerdict:
        return EvaluatorVerdict(evaluator_id="domain", passed=True, score=1.0)

    evaluator = Evaluator([suite(count=1)], evaluators={"extraction": judge})
    report = evaluator.run_iteration(0, SPEC, always(graded=False))
    assert report.domain("extraction").accuracy.require() == 1.0


# --------------------------------------------------------------------------- #
# RULE 2: an empty denominator is UNDEFINED, not zero
# --------------------------------------------------------------------------- #


def test_a_domain_with_no_tasks_reports_undefined_not_zero():
    report = Evaluator([DomainSuite(domain="empty")]).run_iteration(0, SPEC, always())
    domain = report.domain("empty")

    for kind, metric in domain.metrics.items():
        assert not metric.is_defined, f"{kind} should be undefined, not {metric.value}"
        assert metric.value is None
        assert metric.reason
        with pytest.raises(ValueError):
            metric.require()


def test_a_category_with_no_runs_never_contributes_a_zero_to_any_average():
    """The rule that costs you the demo: a silent zero drags the aggregate down."""
    ran = suite("extraction", count=1)
    empty = DomainSuite(domain="never-ran")

    with_empty = Evaluator([ran, empty]).run_iteration(0, SPEC, always(passed=True, tokens=100))
    without_empty = Evaluator([ran]).run_iteration(0, SPEC, always(passed=True, tokens=100))

    for kind in ("accuracy", "reliability", "cost", "speed"):
        included = with_empty.average(kind)
        alone = without_empty.average(kind)
        assert included.require() == pytest.approx(alone.require()), (
            f"the empty category changed the mean {kind}"
        )
        # Averaged over the one domain that ran, not over two.
        assert included.sample_size == 1

    # Had the empty domain contributed a zero, the mean accuracy would be 0.5.
    assert with_empty.mean_accuracy.require() == pytest.approx(1.0)


def test_average_of_all_undefined_metrics_is_undefined_not_zero():
    result = mean_of_defined(
        [undefined("accuracy", "nothing ran"), undefined("accuracy", "nothing ran either")],
        kind="accuracy",
    )
    assert not result.is_defined
    assert result.reason


def test_a_defined_metric_cannot_have_an_empty_denominator():
    with pytest.raises(ValueError, match="undefined, not zero"):
        Metric(kind="accuracy", value=0.0, sample_size=0)


def test_an_undefined_metric_must_carry_a_reason():
    with pytest.raises(ValueError, match="must carry a reason"):
        Metric(kind="accuracy", value=None)


def test_a_defined_metric_may_not_also_carry_an_undefined_reason():
    with pytest.raises(ValueError, match="must not carry"):
        Metric(kind="accuracy", value=1.0, sample_size=3, reason="both at once")


def test_a_genuine_zero_is_still_a_zero():
    """Undefined is not a synonym for zero: a domain that ran and failed scores 0.0."""
    report = Evaluator([suite(count=1)]).run_iteration(0, SPEC, always(passed=False))
    accuracy = report.domain("extraction").accuracy
    assert accuracy.is_defined
    assert accuracy.require() == 0.0


# --------------------------------------------------------------------------- #
# RULE 3: reliability needs >= 3 runs and is a VARIANCE
# --------------------------------------------------------------------------- #


def test_fewer_than_three_repeats_is_rejected_outright():
    for repeats in (0, 1, 2):
        with pytest.raises(ValueError, match="at least 3 runs"):
            Evaluator([suite()], repeats=repeats)


def test_default_repeats_is_three():
    evaluator = Evaluator([suite()])
    assert evaluator.repeats == 3

    calls: list[int] = []

    def runner(spec, task_spec, attempt):
        calls.append(attempt)
        return trajectory(task_spec.task_id)

    evaluator.run_iteration(0, SPEC, runner)
    assert calls == [0, 1, 2, 0, 1, 2]


def test_reliability_is_a_variance_not_a_mean():
    """A perfectly consistent agent has variance 0 even though its mean is 1.0."""
    report = Evaluator([suite(count=1)]).run_iteration(0, SPEC, always(passed=True))
    reliability = report.domain("extraction").reliability

    assert reliability.require() == pytest.approx(0.0)
    assert report.domain("extraction").accuracy.require() == pytest.approx(1.0)


def test_a_flaky_task_has_higher_variance_than_a_consistent_one():
    def flaky(spec, task_spec, attempt):
        return trajectory(task_spec.task_id, passed=attempt == 0)

    flaky_report = Evaluator([suite(count=1)]).run_iteration(0, SPEC, flaky)
    steady_report = Evaluator([suite(count=1)]).run_iteration(0, SPEC, always(passed=False))

    assert flaky_report.domain("extraction").reliability.require() > steady_report.domain(
        "extraction"
    ).reliability.require()
    # 1 pass in 3 -> population variance of [1, 0, 0]
    assert flaky_report.domain("extraction").reliability.require() == pytest.approx(2 / 9)


def test_variance_needs_at_least_two_observations():
    with pytest.raises(ValueError, match="at least two"):
        population_variance([1.0])
    assert population_variance([0.0, 1.0]) == pytest.approx(0.25)


def test_reliability_sample_size_records_the_run_count():
    report = Evaluator([suite(count=1)], repeats=5).run_iteration(0, SPEC, always())
    result = report.domain("extraction").results[0]
    assert result.variance().sample_size == 5


# --------------------------------------------------------------------------- #
# Cost and speed come off the Trajectory
# --------------------------------------------------------------------------- #


def test_cost_and_speed_are_read_from_the_trajectory():
    report = Evaluator([suite(count=2)]).run_iteration(
        0, SPEC, always(tokens=250, seconds=4.0)
    )
    domain = report.domain("extraction")

    assert domain.cost.require() == pytest.approx(250.0)
    assert domain.speed.require() == pytest.approx(4.0)
    assert domain.cost.sample_size == 6


def test_cost_counts_prompt_and_completion_tokens():
    def runner(spec, task_spec, attempt):
        return Trajectory(
            trajectory_id="t",
            spec_id=SPEC.spec_id,
            task_id=task_spec.task_id,
            tokens=TokenUsage(prompt_tokens=30, completion_tokens=70),
            final_answer="42",
            verdict=EvaluatorVerdict(evaluator_id="e", passed=True, score=1.0),
        )

    report = Evaluator([suite(count=1)]).run_iteration(0, SPEC, runner)
    assert report.domain("extraction").cost.require() == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# Harness plumbing
# --------------------------------------------------------------------------- #


def test_suite_rejects_a_task_from_another_domain():
    with pytest.raises(ValueError, match="declares domain"):
        DomainSuite(domain="extraction", tasks=(task("t0", "code-math"),))


def test_suite_rejects_duplicate_task_ids():
    with pytest.raises(ValueError, match="duplicate task_id"):
        DomainSuite(domain="extraction", tasks=(task("t0"), task("t0")))


def test_duplicate_domains_are_rejected():
    with pytest.raises(ValueError, match="duplicate domain"):
        Evaluator([suite("extraction"), suite("extraction")])


def test_every_domain_appears_in_the_iteration_report():
    report = Evaluator([suite("extraction"), suite("code-math")]).run_iteration(
        0, SPEC, always()
    )
    assert {domain.domain for domain in report.domains} == {"extraction", "code-math"}
    with pytest.raises(KeyError):
        report.domain("nope")


def test_a_refused_run_cannot_be_recorded_as_passed():
    from agent_engineer.evaluation import RunOutcome

    with pytest.raises(ValueError, match="cannot be recorded as passed"):
        RunOutcome(task_id="t", attempt=0, passed=True, refused=True)


# --------------------------------------------------------------------------- #
# REPORTING RULE: no improvement without its before number
# --------------------------------------------------------------------------- #


def test_an_improvement_cannot_be_built_without_a_before_number():
    """``before`` is required positionally: there is no after-only constructor."""
    with pytest.raises(TypeError):
        Improvement("accuracy", Metric(kind="accuracy", value=0.9, sample_size=3))  # type: ignore[call-arg]


def test_an_improvement_renders_before_and_after_together():
    rendered = Improvement(
        "accuracy",
        Metric(kind="accuracy", value=0.40, sample_size=6),
        Metric(kind="accuracy", value=0.80, sample_size=6),
    ).render()
    assert "0.400" in rendered and "0.800" in rendered and "+0.400" in rendered


def test_an_improvement_with_an_undefined_before_refuses_to_show_the_after():
    rendered = Improvement(
        "accuracy",
        undefined("accuracy", "nothing ran"),
        Metric(kind="accuracy", value=0.95, sample_size=6),
    ).render()
    assert "0.95" not in rendered
    assert "no before number" in rendered


def test_a_progress_table_needs_two_iterations():
    report = Evaluator([suite()]).run_iteration(0, SPEC, always())
    with pytest.raises(ValueError, match="no before number"):
        render_progress_table([report])


# --------------------------------------------------------------------------- #
# REPORTING RULE: no accuracy without its paired cost
# --------------------------------------------------------------------------- #


def test_accuracy_and_cost_are_rendered_by_a_single_call():
    accuracy_cell, cost_cell = format_accuracy_with_cost(
        Metric(kind="accuracy", value=0.75, sample_size=6),
        Metric(kind="cost", value=1200.0, sample_size=6),
    )
    assert accuracy_cell == "0.750"
    assert "1200" in cost_cell


def test_rendering_accuracy_against_a_non_cost_metric_is_refused():
    with pytest.raises(ValueError, match="without its paired cost"):
        format_accuracy_with_cost(
            Metric(kind="accuracy", value=0.75, sample_size=6),
            Metric(kind="speed", value=1.0, sample_size=6),
        )


def test_every_accuracy_cell_in_a_table_has_a_cost_beside_it():
    report = Evaluator([suite(count=2)]).run_iteration(0, SPEC, always(tokens=500))
    table = render_iteration_table(report)

    header = table.splitlines()[1]
    assert "accuracy" in header and "cost" in header
    assert header.index("accuracy") < header.index("cost")
    assert "500" in table


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def test_an_undefined_metric_renders_as_na_never_as_zero():
    assert format_metric(undefined("accuracy", "nothing ran")) == UNDEFINED_CELL
    assert format_metric(Metric(kind="accuracy", value=0.0, sample_size=3)) == "0.000"


def test_the_table_footnotes_every_undefined_metric_with_its_reason():
    report = Evaluator([suite("extraction"), DomainSuite(domain="never-ran")]).run_iteration(
        0, SPEC, always()
    )
    table = render_iteration_table(report)

    assert UNDEFINED_CELL in table
    assert "never counted as zero" in table
    assert "never-ran.accuracy" in table


def test_the_mean_row_reports_how_many_domains_it_averaged():
    report = Evaluator([suite("extraction"), DomainSuite(domain="never-ran")]).run_iteration(
        0, SPEC, always()
    )
    assert "MEAN (1/2 defined)" in render_iteration_table(report)


def _two_iterations() -> list[IterationReport]:
    evaluator = Evaluator([suite(count=2)])
    before = evaluator.run_iteration(0, SPEC, always(passed=False, tokens=100))
    after = evaluator.run_iteration(1, SPEC, always(passed=True, tokens=400))
    return [before, after]


def test_the_progress_table_pairs_every_accuracy_gain_with_its_cost():
    """Retries buy accuracy. The table must show what the gain cost."""
    table = render_progress_table(_two_iterations())

    assert "0.000 -> 1.000" in table  # accuracy, with its before number
    assert "100.0 tok -> 400.0 tok" in table  # and what it cost to get there
    assert "+300.0" in table


def test_the_full_report_renders_every_iteration_and_the_summary():
    rendered = render_report(_two_iterations())
    assert "iteration 0" in rendered
    assert "iteration 1" in rendered
    assert "progress: iteration 0 -> 1" in rendered


def test_a_report_with_no_iterations_is_refused():
    with pytest.raises(ValueError, match="no iterations were run"):
        render_report([])


def test_a_single_iteration_report_omits_the_progress_section():
    report = Evaluator([suite()]).run_iteration(0, SPEC, always())
    rendered = render_report([report])
    assert "iteration 0" in rendered
    assert "progress:" not in rendered


def test_a_trajectory_for_the_wrong_task_is_failed_not_mis_attributed():
    """Crediting one task's run to another would corrupt every per-task number."""

    def runner(spec, task_spec, attempt):
        return trajectory("t0", passed=True)  # always t0, whatever was asked for

    domain = Evaluator([suite(count=2)]).run_iteration(0, SPEC, runner).domain("extraction")

    assert domain.accuracy.require() == pytest.approx(0.5)
    mismatched = domain.results[1].outcomes[0]
    assert mismatched.errored and not mismatched.passed
    assert "when asked for 't1'" in mismatched.detail
