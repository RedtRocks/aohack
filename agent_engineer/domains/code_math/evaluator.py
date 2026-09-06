"""The code_math evaluator: grade by execution or exact comparison, never by judgement.

Exported as :func:`evaluate`, a plain callable matching the harness's
``TaskEvaluator`` protocol::

    verdict = evaluate(task, trajectory)   # (TaskSpec, Trajectory) -> EvaluatorVerdict

It reads exactly two things off its arguments -- ``task.metadata`` /
``task.expected``, and ``trajectory.final_answer`` -- so it does not care how
the answer was produced, and grades only: no accuracy, cost, or reliability is
computed here, and nothing is aggregated. Counting is the harness's job.

Grading is total and deterministic:

* A **code** task (``metadata["kind"] == "code"``) runs the submission against
  ``metadata["hidden_cases"]`` in a subprocess sandbox (see :mod:`.sandbox`).
  ``score`` is the fraction of cases that passed -- a submission right on five
  of six is visibly better than one right on none, and that gradient is what
  the mutation stage steers by. ``passed`` is all-or-nothing.
* A **math** task compares ``trajectory.final_answer`` to ``task.expected``
  numerically: exactly when ``metadata["tolerance"]`` is 0, within that
  absolute tolerance otherwise. Score is 1.0 or 0.0.

Nothing here can raise on a bad submission. Garbage, an empty answer, an
infinite loop, a process that kills itself -- each comes back as a failed
verdict with a rationale saying which, because a task that hangs must cost the
harness one timeout rather than stall it.
"""

from __future__ import annotations

import json
import math
import re
from decimal import Decimal, InvalidOperation

from agent_engineer.domains.code_math.sandbox import DEFAULT_TIMEOUT_SECONDS, run_python
from agent_engineer.evaluation import TaskSpec
from agent_engineer.schemas import EvaluatorVerdict, Trajectory

__all__ = ["EVALUATOR_ID", "evaluate", "evaluate_answer"]

EVALUATOR_ID = "code_math-exec-v1"

_RESULT_SENTINEL = "@@AE_RESULT@@"
"""Marks the one line of child stdout the harness reads. Candidate code is free to
print whatever it likes; only a line with this prefix is treated as the verdict."""

_FENCE_RE = re.compile(r"^```[^\n]*\n(?P<body>.*?)\n?```\s*$", re.DOTALL)
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _strip_code_fence(answer: str) -> str:
    """Return the body of a fenced block, or the answer unchanged if it is not fenced."""
    match = _FENCE_RE.match(answer.strip())
    return match.group("body") if match else answer


def _normalize_number(text: str) -> Decimal | None:
    """Parse a numeric answer, tolerating separators, currency, and a trailing sentence.

    Returns ``None`` when the text contains no number at all. Lenient about
    *packaging* (``"$1,392.91."``), strict about *value*: never rounds, never
    rescales, never rescues a wrong number.
    """
    cleaned = text.strip().replace("$", "")
    for candidate in (cleaned, *reversed(_NUMBER_RE.findall(cleaned))):
        try:
            return Decimal(candidate.replace(",", "").strip().rstrip("."))
        except (InvalidOperation, ValueError):
            continue
    return None


def _build_harness(entry_point: str, hidden_cases: list[dict[str, str]]) -> str:
    """Emit the child script appended after the candidate's own source."""
    return f'''

def _ae_grade():
    import json
    cases = json.loads({json.dumps(json.dumps(hidden_cases))})
    entry = {entry_point!r}
    outcomes = []
    if entry not in globals() or not callable(globals()[entry]):
        outcomes = [{{"ok": False, "why": "no callable named " + entry}} for _ in cases]
    else:
        for case in cases:
            try:
                actual = eval(case["call"], globals())
                expected = eval(case["expected"], {{}})
                ok = _ae_same(actual, expected)
                why = "" if ok else "got " + repr(actual)[:120]
            except BaseException as exc:
                ok, why = False, type(exc).__name__ + ": " + str(exc)[:120]
            outcomes.append({{"ok": ok, "why": why}})
    print({_RESULT_SENTINEL!r} + json.dumps(outcomes))


def _ae_same(actual, expected):
    if isinstance(actual, bool) != isinstance(expected, bool):
        return False
    if isinstance(actual, float) or isinstance(expected, float):
        try:
            import math as _m
            return _m.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
        except TypeError:
            return False
    try:
        return actual == expected
    except BaseException:
        return False


_ae_grade()
'''


def _parse_outcomes(stdout: str) -> list[dict] | None:
    """Pull the harness verdict off the child's stdout, ignoring anything else it printed."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(_RESULT_SENTINEL):
            try:
                parsed = json.loads(line[len(_RESULT_SENTINEL) :])
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, list) else None
    return None


def _verdict(passed: bool, score: float, rationale: str) -> EvaluatorVerdict:
    return EvaluatorVerdict(
        evaluator_id=EVALUATOR_ID, passed=passed, score=score, rationale=rationale
    )


def _grade_code(
    answer: str, *, entry_point: str, hidden_cases: list[dict[str, str]],
    timeout_seconds: float,
) -> EvaluatorVerdict:
    source = _strip_code_fence(answer).strip()
    if not source:
        return _verdict(False, 0.0, "empty submission")
    try:
        compile(source, "<submission>", "exec")
    except SyntaxError as exc:
        return _verdict(False, 0.0, f"submission does not parse: {exc.msg}")

    result = run_python(
        source + _build_harness(entry_point, hidden_cases), timeout_seconds=timeout_seconds
    )
    total = len(hidden_cases)
    if result.timed_out:
        return _verdict(False, 0.0, f"timed out after {timeout_seconds:g}s on hidden cases")
    outcomes = _parse_outcomes(result.stdout)
    if outcomes is None or len(outcomes) != total:
        detail = (result.stderr.strip().splitlines() or ["no output"])[-1]
        return _verdict(False, 0.0, f"submission failed to run: {detail[:200]}")

    passed_count = sum(1 for outcome in outcomes if outcome.get("ok"))
    first_failure = next(
        (outcome.get("why", "") for outcome in outcomes if not outcome.get("ok")), ""
    )
    rationale = f"{passed_count}/{total} hidden cases passed"
    if first_failure:
        rationale += f"; first failure: {first_failure}"
    return _verdict(passed_count == total, passed_count / total, rationale)


def _grade_math(answer: str, *, expected: str, tolerance: float) -> EvaluatorVerdict:
    submitted = _normalize_number(_strip_code_fence(answer))
    if submitted is None:
        return _verdict(False, 0.0, "no number found in the answer")
    truth = _normalize_number(expected)
    assert truth is not None, f"expected value {expected!r} is not numeric"
    if tolerance == 0.0:
        correct = submitted == truth
    else:
        correct = math.isclose(float(submitted), float(truth), rel_tol=0.0, abs_tol=tolerance)
    return _verdict(correct, 1.0 if correct else 0.0, f"expected {expected}, got {submitted}")


def evaluate_answer(
    task: TaskSpec, answer: str, *, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
) -> EvaluatorVerdict:
    """Grade a raw answer string against a task. Never raises."""
    kind = task.metadata.get("kind")
    if kind == "code":
        return _grade_code(
            answer,
            entry_point=task.metadata["entry_point"],
            hidden_cases=task.metadata["hidden_cases"],
            timeout_seconds=timeout_seconds,
        )
    if kind == "math":
        assert task.expected is not None, f"math task {task.task_id} has no expected value"
        return _grade_math(
            answer, expected=task.expected, tolerance=float(task.metadata.get("tolerance", 0.0))
        )
    raise ValueError(f"task {task.task_id!r} has unknown metadata kind {kind!r}")


def evaluate(task: TaskSpec, trajectory: Trajectory) -> EvaluatorVerdict:
    """The ``TaskEvaluator``: grade one trajectory's final answer against its task.

    Never raises. A missing final answer -- a refusal, a crash the runner
    caught, an unfinished run -- is a failed verdict, not an exception.
    """
    if trajectory.final_answer is None:
        return _verdict(False, 0.0, "no final answer was produced")
    return evaluate_answer(task, trajectory.final_answer)
