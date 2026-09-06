"""Tests for the structured-extraction domain: task set, evaluator, and the
required proof that the engine's harness produces a non-zero baseline score
from this domain's exports alone.

This module imports the domain only through its two public callables,
``get_suite`` and ``get_evaluator``, exactly as the engine is required to.
"""

from __future__ import annotations

import json
import re
from collections import Counter

import pytest

from agent_engineer.domains.extraction import get_evaluator, get_suite
from agent_engineer.evaluation import Evaluator, TaskSpec
from agent_engineer.schemas import AgentSpec, EvaluatorVerdict, TokenUsage, Trajectory

# --------------------------------------------------------------------------- #
# suite structure
# --------------------------------------------------------------------------- #


def test_suite_domain_is_extraction() -> None:
    suite = get_suite()
    assert suite.domain == "extraction"


def test_suite_has_multiple_tasks_with_unique_ids() -> None:
    suite = get_suite()
    ids = [task.task_id for task in suite.tasks]
    assert len(ids) >= 10
    assert len(ids) == len(set(ids))


def test_every_task_declares_the_suite_domain() -> None:
    suite = get_suite()
    assert all(task.domain == suite.domain for task in suite.tasks)


def test_difficulty_is_ranged_not_uniform() -> None:
    """Easy, medium, and hard bands are all present -- a flat task set gives no gradient."""
    suite = get_suite()
    difficulties = Counter(task.metadata["difficulty"] for task in suite.tasks)
    assert difficulties["easy"] > 0
    assert difficulties["medium"] > 0
    assert difficulties["hard"] > 0


def test_some_ground_truth_values_are_deliberately_absent() -> None:
    """A field missing from the document must be gradeable as null, not skipped."""
    suite = get_suite()
    absent = sum(
        1
        for task in suite.tasks
        for value in task.metadata["ground_truth"].values()
        if value is None
    )
    assert absent > 0


def test_get_suite_is_pure() -> None:
    """Two calls produce equal, independently constructed suites."""
    a, b = get_suite(), get_suite()
    assert a is not b
    assert [t.task_id for t in a.tasks] == [t.task_id for t in b.tasks]


# --------------------------------------------------------------------------- #
# evaluator: field-level grading
# --------------------------------------------------------------------------- #


def _task_by_id(task_id: str) -> TaskSpec:
    suite = get_suite()
    for task in suite.tasks:
        if task.task_id == task_id:
            return task
    raise AssertionError(f"no such task {task_id!r}")


def _trajectory_for(task: TaskSpec, answer: dict | str | None, task_id: str | None = None) -> Trajectory:
    final_answer = answer if isinstance(answer, str) or answer is None else json.dumps(answer)
    return Trajectory(
        trajectory_id=f"traj-{task.task_id}",
        spec_id="spec-under-test",
        task_id=task_id or task.task_id,
        final_answer=final_answer,
    )


def test_exact_ground_truth_scores_perfectly() -> None:
    task = _task_by_id("extract-easy-invoice-001")
    trajectory = _trajectory_for(task, dict(task.metadata["ground_truth"]))
    verdict = get_evaluator()(task, trajectory)
    assert verdict.passed is True
    assert verdict.score == pytest.approx(1.0)


def test_wrong_values_fail_and_lower_the_score() -> None:
    task = _task_by_id("extract-easy-invoice-001")
    answer = dict(task.metadata["ground_truth"])
    answer["customer_name"] = "Some Other Company"
    verdict = get_evaluator()(task, _trajectory_for(task, answer))
    assert verdict.passed is False
    assert 0.0 < verdict.score < 1.0


def test_no_final_answer_scores_zero() -> None:
    task = _task_by_id("extract-easy-invoice-001")
    verdict = get_evaluator()(task, _trajectory_for(task, None))
    assert verdict.passed is False
    assert verdict.score == 0.0


def test_non_json_final_answer_scores_zero() -> None:
    task = _task_by_id("extract-easy-invoice-001")
    verdict = get_evaluator()(task, _trajectory_for(task, "sure, the invoice number is INV-1"))
    assert verdict.passed is False
    assert verdict.score == 0.0


def test_json_fenced_in_a_code_block_is_still_graded() -> None:
    task = _task_by_id("extract-easy-invoice-001")
    fenced = "```json\n" + json.dumps(dict(task.metadata["ground_truth"])) + "\n```"
    verdict = get_evaluator()(task, _trajectory_for(task, fenced))
    assert verdict.passed is True
    assert verdict.score == pytest.approx(1.0)


def test_absent_ground_truth_accepts_null_or_a_stated_absence() -> None:
    task = _task_by_id("extract-medium-ocr-invoice-004")
    assert task.metadata["ground_truth"]["purchase_order_number"] is None
    answer = dict(task.metadata["ground_truth"])
    answer["purchase_order_number"] = "N/A"
    verdict = get_evaluator()(task, _trajectory_for(task, answer))
    assert verdict.passed is True


def test_absent_ground_truth_rejects_an_invented_value() -> None:
    task = _task_by_id("extract-medium-ocr-invoice-004")
    answer = dict(task.metadata["ground_truth"])
    answer["purchase_order_number"] = "PO-99999"
    verdict = get_evaluator()(task, _trajectory_for(task, answer))
    assert verdict.passed is False


def test_money_normalizes_currency_formatting() -> None:
    task = _task_by_id("extract-easy-invoice-001")
    answer = dict(task.metadata["ground_truth"])
    answer["total_due"] = "$1,291.14"
    verdict = get_evaluator()(task, _trajectory_for(task, answer))
    assert verdict.passed is True


def test_date_normalizes_alternate_spellings() -> None:
    task = _task_by_id("extract-easy-invoice-001")
    answer = dict(task.metadata["ground_truth"])
    answer["invoice_date"] = "March 14, 2024"
    verdict = get_evaluator()(task, _trajectory_for(task, answer))
    assert verdict.passed is True


def test_list_field_ignores_order_and_delimiter_style() -> None:
    task = _task_by_id("extract-easy-lab-report-003")
    answer = dict(task.metadata["ground_truth"])
    answer["panels_performed"] = "Lipid Panel; COMPLETE BLOOD COUNT ;basic metabolic panel"
    verdict = get_evaluator()(task, _trajectory_for(task, answer))
    assert verdict.passed is True


def test_hard_task_penalizes_the_superseded_value() -> None:
    """The header figures are explicitly superseded; using them must not pass."""
    task = _task_by_id("extract-hard-superseded-statement-009")
    answer = dict(task.metadata["ground_truth"])
    answer["closing_balance"] = "1204880.15"  # the superseded header figure
    answer["statement_date"] = "2024-07-31"  # the superseded header date
    verdict = get_evaluator()(task, _trajectory_for(task, answer))
    assert verdict.passed is False


def test_evaluator_rejects_a_trajectory_for_a_different_task() -> None:
    task = _task_by_id("extract-easy-invoice-001")
    other = _task_by_id("extract-easy-purchase-order-002")
    trajectory = _trajectory_for(other, dict(other.metadata["ground_truth"]))
    with pytest.raises(ValueError):
        get_evaluator()(task, trajectory)


# --------------------------------------------------------------------------- #
# definition of done: the harness, given only this domain's exports, produces
# a non-zero baseline score.
# --------------------------------------------------------------------------- #

_DOC_BOUNDS = re.compile(r"--- BEGIN DOCUMENT ---\n(.*)\n--- END DOCUMENT ---", re.DOTALL)
_LABEL_LINE = re.compile(r"^\s*([A-Za-z][\w /&'.\-]{1,40}?)\s*(?:\.{2,}|:)\s*(.+?)\s*$")


def _naive_label_scan(document: str) -> dict[str, str]:
    """A weak, generic 'read labelled lines' strategy. No access to ground truth."""
    labels: dict[str, str] = {}
    for line in document.splitlines():
        match = _LABEL_LINE.match(line)
        if not match:
            continue
        label = re.sub(r"[^a-z0-9 ]", "", match.group(1).strip().lower())
        value = match.group(2).strip()
        if label and value:
            labels.setdefault(label, value)
    return labels


def _best_label_match(field_name: str, labels: dict[str, str]) -> str | None:
    tokens = set(field_name.replace("_", " ").split())
    if not tokens:
        return None
    best_value, best_score = None, 0.0
    for label, value in labels.items():
        score = len(tokens & set(label.split())) / len(tokens)
        if score > best_score:
            best_score, best_value = score, value
    return best_value if best_score >= 0.5 else None


def _baseline_runner(spec: AgentSpec, task: TaskSpec, attempt: int) -> Trajectory:
    """A domain-naive baseline agent: scans for labelled lines, nothing smarter.

    It never reads ``ground_truth``. It clears the easy band by pattern-matching
    field names to document labels, and is expected to fail most of the hard
    band, which is exactly the gradient the optimization loop needs.
    """
    match = _DOC_BOUNDS.search(task.prompt)
    document = match.group(1) if match else task.prompt
    labels = _naive_label_scan(document)
    answer = {
        field["name"]: _best_label_match(field["name"], labels)
        for field in task.metadata["fields"]
    }
    return Trajectory(
        trajectory_id=f"traj-{task.task_id}-{attempt}",
        spec_id=spec.spec_id,
        task_id=task.task_id,
        tokens=TokenUsage(prompt_tokens=len(task.prompt.split()), completion_tokens=20),
        elapsed_seconds=0.01,
        final_answer=json.dumps(answer),
    )


def test_baseline_runner_produces_a_non_zero_accuracy_through_the_real_harness() -> None:
    """The definition of done: the engine's harness, given only this domain's
    `get_suite()` and `get_evaluator()`, scores a naive baseline above zero.
    """
    suite = get_suite()
    harness = Evaluator(
        suites=[suite],
        evaluators={suite.domain: get_evaluator()},
        repeats=3,
    )
    spec = AgentSpec(spec_id="baseline-spec", system_prompt="Extract the requested fields.")

    report = harness.run_iteration(0, spec, _baseline_runner)
    domain_report = report.domain("extraction")

    assert domain_report.accuracy.value is not None
    assert domain_report.accuracy.value > 0.0, "baseline must clear the floor the easy band sets"
    assert domain_report.accuracy.value < 1.0, "baseline must leave headroom for the hard band"


def test_baseline_is_deterministic_across_runs() -> None:
    """The naive runner has no randomness, so repeats agree and reliability is defined."""
    suite = get_suite()
    harness = Evaluator(
        suites=[suite],
        evaluators={suite.domain: get_evaluator()},
        repeats=3,
    )
    spec = AgentSpec(spec_id="baseline-spec", system_prompt="Extract the requested fields.")

    report = harness.run_iteration(0, spec, _baseline_runner)
    domain_report = report.domain("extraction")

    assert domain_report.reliability.value is not None
    assert domain_report.reliability.value == pytest.approx(0.0)
