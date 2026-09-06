"""The deterministic evaluator for the structured-extraction task set.

Given a :class:`~agent_engineer.evaluation.TaskSpec` and the
:class:`~agent_engineer.schemas.Trajectory` run against it, this scores the
agent's ``final_answer`` against the ground truth carried in
``task.metadata`` and returns an :class:`~agent_engineer.schemas.EvaluatorVerdict`.

It is deterministic end to end: no model call, no network, no clock, no
randomness. The same (task, trajectory) pair always scores the same. The only
judgement it makes is normalization -- ``'$1,234.50'`` and ``'1234.5'`` are the
same money, ``'Jan 5, 2024'`` and ``'2024-01-05'`` are the same date -- and that
normalization is fixed in this file, not inferred per run.

Scoring is per field: the score is the fraction of the task's fields the agent
got right, and the verdict passes when that fraction reaches the task's
``pass_threshold``. Two whole-output failures short-circuit to zero, and both
are recorded in the rationale in a form the diagnosis stage can read:

* no ``final_answer`` at all -- the run produced nothing to grade;
* a ``final_answer`` that is not a JSON object with the required keys -- the
  agent's work may have been right but the contract it was given was not met.

The rationale is a stable, line-oriented report naming every field and why it
failed, so a diagnosis stage has something concrete to attribute a
:class:`~agent_engineer.schemas.FailureCause` to.

This module grades only: it never computes accuracy, cost, speed, or
reliability, and never aggregates across tasks -- counting is the harness's
job (:mod:`agent_engineer.evaluation`). The engine must never import it
directly; it reaches this domain only through ``get_suite()`` and
``get_evaluator()`` in :mod:`agent_engineer.domains.extraction`.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from agent_engineer.domains.extraction.tasks import FieldKind
from agent_engineer.evaluation import TaskSpec
from agent_engineer.schemas import EvaluatorVerdict, Trajectory

__all__ = [
    "EVALUATOR_ID",
    "FieldOutcome",
    "ExtractionEvaluator",
    "EVALUATOR",
    "evaluate",
]

EVALUATOR_ID = "structured-extraction-v1"

_ABSENT_SYNONYMS = frozenset(
    {
        "",
        "-",
        "--",
        "n/a",
        "na",
        "none",
        "null",
        "nil",
        "not provided",
        "not stated",
        "not given",
        "not present",
        "not specified",
        "unknown",
        "absent",
    }
)
"""Spellings of "the document does not say" that count as a correct ``null``.

An agent that writes ``"N/A"`` where the ground truth is ``None`` has read the
document correctly and formatted the answer loosely. Failing it there would
punish the wrong thing and cost the loop a real signal.
"""

_DATE_FORMATS = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%Y/%m/%d",
    "%d %B %Y",
    "%d %b %Y",
    "%B %d %Y",
    "%b %d %Y",
    "%d-%b-%Y",
)

_ORDINAL_SUFFIX = re.compile(r"(?<=\d)(st|nd|rd|th)\b", re.IGNORECASE)
_LIST_SPLIT = re.compile(r"[,;|\n]+")
_WHITESPACE = re.compile(r"\s+")
_MONEY_STRIP = re.compile(r"[^\d.\-]")


# --------------------------------------------------------------------------- #
# Normalization. Each kind gets one canonical form; equality is on that form.
# --------------------------------------------------------------------------- #


def _collapse(value: str) -> str:
    """Casefold, normalize unicode, collapse whitespace, drop edge punctuation."""
    text = unicodedata.normalize("NFKD", value)
    text = _WHITESPACE.sub(" ", text).strip()
    text = text.strip(" .,:;\"'")
    return text.casefold()


def _is_absent(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return _collapse(value) in _ABSENT_SYNONYMS
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    return False


def _normalize_money(value: Any) -> Decimal | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        return Decimal(str(value)).quantize(Decimal("0.01"))
    if not isinstance(value, str):
        return None
    cleaned = _MONEY_STRIP.sub("", value.replace(",", ""))
    if cleaned in {"", "-", "."}:
        return None
    try:
        return Decimal(cleaned).quantize(Decimal("0.01"))
    except (InvalidOperation, ArithmeticError):
        return None


def _normalize_integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if not isinstance(value, str):
        return None
    cleaned = value.replace(",", "").replace("_", "").strip()
    cleaned = re.sub(r"\s*(kwh|units|drums|pieces|pcs)\s*$", "", cleaned, flags=re.IGNORECASE)
    try:
        return int(cleaned)
    except ValueError:
        return None


def _normalize_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    text = _ORDINAL_SUFFIX.sub("", value).replace(",", " ")
    text = _WHITESPACE.sub(" ", text).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _normalize_list(value: Any) -> frozenset[str] | None:
    if isinstance(value, str):
        items: list[str] = _LIST_SPLIT.split(value)
    elif isinstance(value, (list, tuple)):
        items = []
        for item in value:
            if isinstance(item, str):
                items.append(item)
            elif item is None:
                continue
            else:
                items.append(str(item))
    else:
        return None
    normalized = {_collapse(item) for item in items}
    normalized.discard("")
    return frozenset(normalized)


def _values_match(kind: FieldKind, expected: Any, actual: Any) -> bool:
    """Compare one extracted value against ground truth under the field's kind."""
    if _is_absent(expected):
        return _is_absent(actual)
    if _is_absent(actual):
        return False

    match kind:
        case FieldKind.MONEY:
            expected_money = _normalize_money(expected)
            actual_money = _normalize_money(actual)
            return expected_money is not None and expected_money == actual_money
        case FieldKind.INTEGER:
            expected_int = _normalize_integer(expected)
            actual_int = _normalize_integer(actual)
            return expected_int is not None and expected_int == actual_int
        case FieldKind.DATE:
            expected_date = _normalize_date(expected)
            actual_date = _normalize_date(actual)
            return expected_date is not None and expected_date == actual_date
        case FieldKind.LIST:
            expected_set = _normalize_list(expected)
            actual_set = _normalize_list(actual)
            return expected_set is not None and expected_set == actual_set
        case FieldKind.ENUM | FieldKind.TEXT:
            if not isinstance(actual, (str, int, float)):
                return False
            return _collapse(str(expected)) == _collapse(str(actual))

    raise AssertionError(f"unhandled field kind {kind!r}")


# --------------------------------------------------------------------------- #
# The evaluator.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class FieldOutcome:
    """The result for one field. The unit the rationale is built from."""

    name: str
    kind: FieldKind
    expected: Any
    actual: Any
    correct: bool
    reason: str

    def render(self) -> str:
        mark = "PASS" if self.correct else "FAIL"
        return f"  [{mark}] {self.name} ({self.kind.value}): {self.reason}"


def _extract_json_object(answer: str) -> dict[str, Any] | None:
    """Parse the agent's answer into a JSON object, or return None.

    A fenced code block is unwrapped first: models emit ```json around
    otherwise-correct output constantly, and treating that as a total failure
    would flatten the score distribution and cost the loop its signal. Anything
    beyond that -- prose around the object, a JSON array, a bare scalar -- is a
    genuine contract violation and is not rescued.
    """
    text = answer.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            body = lines[1:]
            if body and body[-1].strip().startswith("```"):
                body = body[:-1]
            text = "\n".join(body).strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


class ExtractionEvaluator:
    """Scores structured-extraction trajectories against ground truth in ``task.metadata``.

    Matches the :class:`~agent_engineer.evaluation.TaskEvaluator` protocol:
    ``(task: TaskSpec, trajectory: Trajectory) -> EvaluatorVerdict``. Grades
    only -- it never aggregates or averages, that is the harness's job.
    """

    evaluator_id: str = EVALUATOR_ID

    def __call__(self, task: TaskSpec, trajectory: Trajectory) -> EvaluatorVerdict:
        return self.evaluate(task, trajectory)

    def evaluate(self, task: TaskSpec, trajectory: Trajectory) -> EvaluatorVerdict:
        """Grade one trajectory. Total function: never raises on bad agent output."""
        if task.task_id != trajectory.task_id:
            raise ValueError(
                f"task {task.task_id!r} does not match trajectory task "
                f"{trajectory.task_id!r}"
            )

        meta = task.metadata
        fields: list[dict[str, Any]] = meta["fields"]
        ground_truth: dict[str, Any] = meta["ground_truth"]
        pass_threshold: float = meta["pass_threshold"]
        difficulty: str = meta["difficulty"]
        field_names = [f["name"] for f in fields]

        if trajectory.final_answer is None:
            return self._zero(
                task, difficulty, len(fields), "no_final_answer: the run produced no final answer"
            )

        payload = _extract_json_object(trajectory.final_answer)
        if payload is None:
            return self._zero(
                task,
                difficulty,
                len(fields),
                "output_format_violation: final answer is not a JSON object",
            )

        outcomes = [
            self._grade_field(field, ground_truth, payload) for field in fields
        ]
        correct = sum(1 for outcome in outcomes if outcome.correct)
        score = correct / len(outcomes)
        passed = score >= pass_threshold

        missing_keys = sorted(set(field_names) - payload.keys())
        extra_keys = sorted(payload.keys() - set(field_names))
        header = [
            f"task={task.task_id} difficulty={difficulty} "
            f"fields={correct}/{len(outcomes)} score={score:.4f} "
            f"threshold={pass_threshold:.2f} passed={str(passed).lower()}"
        ]
        if missing_keys:
            header.append(f"  missing keys: {', '.join(missing_keys)}")
        if extra_keys:
            header.append(f"  unexpected keys: {', '.join(extra_keys)}")
        rationale = "\n".join(header + [outcome.render() for outcome in outcomes])

        return EvaluatorVerdict(
            evaluator_id=self.evaluator_id,
            passed=passed,
            score=round(score, 6),
            rationale=rationale,
        )

    def _grade_field(
        self, field: dict[str, Any], ground_truth: dict[str, Any], payload: dict[str, Any]
    ) -> FieldOutcome:
        name = field["name"]
        kind = FieldKind(field["kind"])
        expected = ground_truth[name]
        if name not in payload:
            return FieldOutcome(
                name=name,
                kind=kind,
                expected=expected,
                actual=None,
                correct=False,
                reason="key missing from output",
            )

        actual = payload[name]
        correct = _values_match(kind, expected, actual)
        if correct:
            reason = "absent, correctly reported as null" if _is_absent(expected) else "matched"
        elif _is_absent(expected):
            reason = f"expected null (field is absent from the document), got {actual!r}"
        elif _is_absent(actual):
            reason = f"expected {expected!r}, got no value"
        else:
            reason = f"expected {expected!r}, got {actual!r}"

        return FieldOutcome(
            name=name,
            kind=kind,
            expected=expected,
            actual=actual,
            correct=correct,
            reason=reason,
        )

    def _zero(
        self, task: TaskSpec, difficulty: str, field_count: int, reason: str
    ) -> EvaluatorVerdict:
        return EvaluatorVerdict(
            evaluator_id=self.evaluator_id,
            passed=False,
            score=0.0,
            rationale=(
                f"task={task.task_id} difficulty={difficulty} "
                f"fields=0/{field_count} score=0.0000 passed=false\n  {reason}"
            ),
        )


EVALUATOR = ExtractionEvaluator()
"""The structured-extraction evaluator. One of this domain's two exports."""


def evaluate(task: TaskSpec, trajectory: Trajectory) -> EvaluatorVerdict:
    """Module-level convenience wrapper around :data:`EVALUATOR`."""
    return EVALUATOR.evaluate(task, trajectory)
