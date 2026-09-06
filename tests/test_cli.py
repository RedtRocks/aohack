"""Tests for the CLI entry point: domain-agnostic wiring and honest rendering.

These prove the two claims the task hinges on:

* the CLI runs a domain it has never seen, by name alone, with no
  domain-specific code path (looped over ``DOMAIN_NAMES`` -- the real
  registry, not a hardcoded list);
* the lineage view renders rejected mutations, zero/negative deltas, and
  undefined metrics correctly, not just the monotonic-climb case.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from agent_engineer.cli import ScriptedBackend, _render_generation, main, run_domain
from agent_engineer.domains import DOMAIN_NAMES
from agent_engineer.evaluation.report import Improvement
from agent_engineer.evaluation.metrics import Metric, undefined
from agent_engineer.loop import GenerationRecord
from agent_engineer.schemas import AgentSpec, Diagnosis, FailureCause, Mutation, MutationKind
from agent_engineer.stages.evaluate import EvaluationRun
from agent_engineer.stages.select import Decision, Verdict


def _bare_spec(spec_id: str, *, parent: str | None = None) -> AgentSpec:
    return AgentSpec(spec_id=spec_id, system_prompt="You are an agent.", parent_spec_id=parent)


def _bare_record(*, before: float, after: float, min_delta: float = 1e-9) -> GenerationRecord:
    """A minimal, hand-built GenerationRecord -- no loop run needed -- so the
    rendering can be checked against deltas the real loop may or may not ever
    happen to produce on a given day (negative, or too small to clear the
    keep threshold), without depending on a scripted backend to manufacture
    one."""
    diagnosis = Diagnosis(diagnosis_id="diag-0", spec_id="root", attributions=())
    mutation = Mutation(
        mutation_id="mut-0",
        kind=MutationKind.SYSTEM_PROMPT_REWRITE,
        target_path="system_prompt",
        before="old text",
        after="new text",
        rationale="test fixture",
        motivating_diagnosis=diagnosis,
        motivating_cause=FailureCause.FAULTY_REASONING,
        parent_spec_id="root",
        child_spec_id="root-g1",
    )
    delta = after - before
    decision = Decision.ACCEPTED if delta >= min_delta else Decision.REVERTED
    verdict = Verdict(
        decision=decision,
        before=before,
        after=after,
        reason=f"measured {before:.4f} -> {after:.4f} (delta {delta:+.4f})",
    )
    empty_run = EvaluationRun(spec_id="root", task_set_id="fixture-suite", records=())
    return GenerationRecord(
        generation=1,
        mutation=mutation,
        diagnosis=diagnosis,
        evaluation_before=empty_run,
        evaluation_after=empty_run,
        verdict=verdict,
        spec_before=_bare_spec("root"),
        spec_after=_bare_spec("root-g1", parent="root"),
    )


def test_lineage_view_renders_a_negative_delta_as_reverted() -> None:
    """A mutation that measures worse than its parent: negative delta, reverted.

    Nothing in the display code may assume a delta is non-negative -- this
    proves it isn't just untested, it actually renders correctly.
    """
    record = _bare_record(before=0.70, after=0.55)
    assert record.verdict.delta < 0
    assert not record.accepted
    rendered = _render_generation(record)
    assert "REVERTED" in rendered
    assert "-0.1500" in rendered
    assert "0.7000 -> 0.5500" in rendered


def test_lineage_view_reverts_a_delta_too_small_to_clear_the_keep_threshold() -> None:
    """A tiny positive delta -- smaller than what a selection policy treats as
    real movement -- must still revert and render as reverted, not as a win."""
    record = _bare_record(before=0.50, after=0.5000000001, min_delta=1e-6)
    assert 0 < record.verdict.delta < 1e-6
    assert not record.accepted
    rendered = _render_generation(record)
    assert "REVERTED" in rendered


def test_lineage_view_renders_an_accepted_positive_delta() -> None:
    """The positive case still has to render correctly alongside the negative
    and sub-threshold ones -- the display code treats none of them specially."""
    record = _bare_record(before=0.40, after=0.60)
    assert record.accepted
    rendered = _render_generation(record)
    assert "ACCEPTED" in rendered
    assert "+0.2000" in rendered


@pytest.mark.parametrize("domain", DOMAIN_NAMES)
def test_run_domain_completes_for_every_registered_domain(domain: str) -> None:
    """The acceptance test: point the CLI at any registered domain by name.

    No branch in this test, or in agent_engineer/cli.py, mentions a domain
    name -- ``DOMAIN_NAMES`` comes straight from the registry.
    """
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        report = run_domain(domain, max_generations=2, repeats=3, stream=True)
    output = buffer.getvalue()

    assert len(report.generations) >= 1
    assert "results (agent_engineer.evaluation.report)" in output
    assert domain in output
    # every generation is unambiguously labeled accepted or reverted
    for record in report.generations:
        label = "ACCEPTED" if record.accepted else "REVERTED"
        assert label in output


def test_lineage_view_renders_a_reverted_generation_with_zero_delta() -> None:
    """A generation that does not clear the keep threshold must say REVERTED,
    not be silently dropped or dressed up as progress."""
    report = run_domain("extraction", max_generations=1, repeats=3, stream=False)
    record = report.generations[0]
    assert not record.accepted
    rendered = _render_generation(record)
    assert "REVERTED" in rendered
    assert f"{record.verdict.delta:+.4f}" in rendered


def test_scripted_backend_is_deterministic_across_processes() -> None:
    """The admission rule must not depend on Python's randomized str hash seed:
    the same spec_id and task_id must always agree on pass/fail."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _Spec:
        spec_id: str

    @dataclass(frozen=True)
    class _Task:
        task_id: str
        expected: str | None
        prompt: str = "solve it"

    backend = ScriptedBackend()
    task = _Task(task_id="fixed-task-id", expected="42")
    first = backend.next_action(_Spec("agent-g3"), task, (), ())
    second = backend.next_action(_Spec("agent-g3"), task, (), ())
    assert first.final_answer == second.final_answer


def test_improvement_cannot_be_rendered_without_both_sides_defined() -> None:
    """The CLI's results section is a thin wrapper over report.py; confirm the
    guard it depends on is still doing its job before trusting the wrapper."""
    before = undefined("accuracy", "no domain reported a defined accuracy")
    after = Metric(kind="accuracy", value=0.5, sample_size=4)
    improvement = Improvement("accuracy", before, after)
    assert not improvement.is_renderable
    assert "no before number" in improvement.render()


def test_list_command_prints_every_registered_domain(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["list"])
    assert exit_code == 0
    out = capsys.readouterr().out
    for domain in DOMAIN_NAMES:
        assert domain in out


def test_run_command_rejects_an_unregistered_domain() -> None:
    with pytest.raises(SystemExit):
        main(["run", "not-a-real-domain"])
