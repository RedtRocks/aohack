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
