"""Tests for the code_math domain: the task suite, the evaluator, and the seam.

The definition of done for this domain is
:func:`test_engine_gets_a_nonzero_baseline_score`: the real, domain-agnostic
``agent_engineer.evaluation.Evaluator`` harness, given only this suite and this
evaluator plus a runner that never looks at ground truth, reports an accuracy
strictly between 0 and 1.
"""

from __future__ import annotations

import pytest

from agent_engineer.domains.code_math import DOMAIN, get_evaluator, get_suite
from agent_engineer.domains.code_math.evaluator import evaluate_answer
from agent_engineer.domains.code_math.sandbox import run_python
from agent_engineer.evaluation import DomainSuite, Evaluator, TaskEvaluator, TaskSpec
from agent_engineer.schemas import AgentSpec, EvaluatorVerdict, Trajectory

SUITE: DomainSuite = get_suite()
EVALUATE: TaskEvaluator = get_evaluator()

SPEC = AgentSpec(spec_id="code-math-baseline", system_prompt="Answer the task.")


def _trajectory(task_id: str, answer: str | None) -> Trajectory:
    return Trajectory(
        trajectory_id=f"t-{task_id}", spec_id=SPEC.spec_id, task_id=task_id,
        final_answer=answer,
    )


# --------------------------------------------------------------------------- #
# The definition of done: the real harness gets a usable signal
# --------------------------------------------------------------------------- #


def _baseline_agent(prompt: str) -> str | None:
    """A deliberately mediocre stand-in: right sometimes, sloppy often, silent mostly.

    Keyed off prompt text only, exactly as a real agent would be -- it never
    sees a task id, a hidden case, or ground truth. Some answers are correct,
    some walk straight into the task's documented trap, the rest go unanswered,
    which is what puts the score strictly between 0 and 1.
    """
    if "count_multiples" in prompt:
        return "def count_multiples(lo, hi, k):\n    return hi // k - (lo - 1) // k\n"
    if "divisible by" in prompt:
        return "233168"
    if "median(nums)" in prompt:
        return "def median(nums):\n    return float(nums[len(nums) // 2])\n"  # trap
    if "roman_to_int" in prompt:
        return (
            "def roman_to_int(s):\n"
            "    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n"
            "    return sum(values[c] for c in s)\n"
        )  # trap
    if "72 km/h" in prompt:
        return "18"  # trap: kilometres, not metres
    return None


def _runner(spec: AgentSpec, task: TaskSpec, attempt: int) -> Trajectory:
    answer = _baseline_agent(task.prompt)
    return Trajectory(
        trajectory_id=f"{spec.spec_id}-{task.task_id}-{attempt}",
        spec_id=spec.spec_id,
        task_id=task.task_id,
        final_answer=answer,
    )


def test_engine_gets_a_nonzero_baseline_score() -> None:
    """A domain-agnostic engine, given only the suite and the evaluator, gets signal."""
    evaluator = Evaluator([SUITE], evaluators={DOMAIN: EVALUATE}, repeats=3)
    report = evaluator.run_iteration(0, SPEC, _runner)

    domain_report = report.domain(DOMAIN)
    accuracy = domain_report.accuracy.require()
    assert accuracy > 0.0, "an all-fail suite gives the engine no gradient to follow"
    assert accuracy < 1.0, "an all-pass suite gives the engine nothing to improve"
    assert domain_report.task_count == len(SUITE.tasks)
    # Reliability is defined because repeats >= 3 and the baseline is deterministic.
    assert domain_report.reliability.value is not None


def test_engine_never_imports_the_domain() -> None:
    """The harness module the engine depends on must not import this domain package."""
    import sys

    for name in ("agent_engineer.evaluation", "agent_engineer.evaluation.harness"):
        module = sys.modules.get(name)
        assert module is not None
        source = getattr(module, "__file__", "") or ""
        assert "code_math" not in source
    assert "agent_engineer.domains" not in vars(sys.modules["agent_engineer.evaluation.harness"])


# --------------------------------------------------------------------------- #
# The suite
# --------------------------------------------------------------------------- #


def test_suite_domain_and_task_ids_are_consistent() -> None:
    assert SUITE.domain == DOMAIN
    ids = [task.task_id for task in SUITE.tasks]
    assert len(set(ids)) == len(ids)
    assert all(task.domain == DOMAIN for task in SUITE.tasks)


def test_difficulty_and_kind_both_span_a_range() -> None:
    """All-easy or all-code would collapse the signal the engine is steering by."""
    difficulties = {task.metadata["difficulty"] for task in SUITE.tasks}
    assert difficulties == {"easy", "medium", "hard"}
    kinds = {task.metadata["kind"] for task in SUITE.tasks}
    assert kinds == {"code", "math"}
    assert len(SUITE.tasks) >= 12


def test_prompt_leaks_no_ground_truth_or_trap() -> None:
    """The prompt is everything the agent sees. It must not contain the answer or the trap note."""
    for task in SUITE.tasks:
        assert task.metadata["trap"] not in task.prompt
        if task.metadata["kind"] == "math":
            assert task.expected not in task.prompt
        else:
            assert task.expected not in task.prompt  # reference_solution
            hidden_only = [
                case for case in task.metadata["hidden_cases"] if case["call"] not in task.prompt
            ]
            assert hidden_only, "every hidden case is visible in the prompt; nothing is hidden"


@pytest.mark.parametrize("task", SUITE.tasks, ids=[task.task_id for task in SUITE.tasks])
def test_reference_answer_passes(task: TaskSpec) -> None:
    """The suite is self-checking: ground truth must score a clean pass through the evaluator."""
    verdict = evaluate_answer(task, task.expected)
    assert verdict.passed, verdict.rationale
    assert verdict.score == 1.0


# --------------------------------------------------------------------------- #
# Near-miss traps: the plausible wrong answer must actually fail
# --------------------------------------------------------------------------- #

_TRAP_ANSWERS: dict[str, str] = {
    "code-inclusive-multiples": "def count_multiples(lo, hi, k):\n"
    "    return (hi - 1) // k - (lo - 1) // k\n",
    "code-median": "def median(nums):\n"
    "    ordered = sorted(nums)\n"
    "    return float(ordered[len(ordered) // 2])\n",
    "code-roman-to-int": "def roman_to_int(s):\n"
    "    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n"
    "    return sum(values[c] for c in s)\n",
    "code-merge-intervals": "def merge_intervals(intervals):\n"
    "    merged = []\n"
    "    for start, end in sorted(intervals):\n"
    "        if merged and start < merged[-1][1]:\n"
    "            merged[-1][1] = max(merged[-1][1], end)\n"
    "        else:\n"
    "            merged.append([start, end])\n"
    "    return merged\n",
    "code-first-index": "def first_index(arr, target):\n"
    "    lo, hi = 0, len(arr) - 1\n"
    "    while lo <= hi:\n"
    "        mid = (lo + hi) // 2\n"
    "        if arr[mid] == target:\n"
    "            return mid\n"
    "        if arr[mid] < target:\n"
    "            lo = mid + 1\n"
    "        else:\n"
    "            hi = mid - 1\n"
    "    return -1\n",
    "code-top-words": "import re\n"
    "from collections import Counter\n"
    "\n"
    "def top_words(text, n):\n"
    "    return Counter(re.findall(r'[A-Za-z]+', text.lower())).most_common(n)\n",
    "code-max-concurrent": "def max_concurrent(meetings):\n"
    "    events = []\n"
    "    for start, end in meetings:\n"
    "        events.append((start, 0))\n"
    "        events.append((end, 1))\n"
    "    events.sort()\n"
    "    best = running = 0\n"
    "    for _, kind in events:\n"
    "        running += 1 if kind == 0 else -1\n"
    "        best = max(best, running)\n"
    "    return best\n",
    "code-min-coins": "def min_coins(coins, amount):\n"
    "    count = 0\n"
    "    for coin in sorted(coins, reverse=True):\n"
    "        while amount >= coin:\n"
    "            amount -= coin\n"
    "            count += 1\n"
    "    return count if amount == 0 else -1\n",
    "math-train-metres": "18",
    "math-divisible-3-or-5": "266333",
    "math-modular-power": "9",
    "math-exactly-two-heads": "0.25",
    "math-compound-quarterly": "1380.00",
    "math-distinct-even-4digit": "2520",
    "math-average-speed": "45.00",
    "math-pool-path-width": "5.464",
}


def test_every_task_has_a_trap_case() -> None:
    assert set(_TRAP_ANSWERS) == {task.task_id for task in SUITE.tasks}


@pytest.mark.parametrize("task_id", sorted(_TRAP_ANSWERS), ids=sorted(_TRAP_ANSWERS))
def test_near_miss_answers_are_rejected(task_id: str) -> None:
    """A plausible-but-wrong attempt must fail. Otherwise the task grades nothing."""
    task = next(t for t in SUITE.tasks if t.task_id == task_id)
    verdict = evaluate_answer(task, _TRAP_ANSWERS[task_id])
    assert not verdict.passed, f"{task_id} accepted its own documented near-miss"


def test_a_near_miss_still_earns_partial_credit() -> None:
    """Code scoring is graded, not binary: mostly-right must outrank all-wrong."""
    task = next(t for t in SUITE.tasks if t.task_id == "code-min-coins")
    near = evaluate_answer(task, _TRAP_ANSWERS["code-min-coins"])
    hopeless = evaluate_answer(task, "def min_coins(coins, amount):\n    return 42\n")
    assert 0.0 < near.score < 1.0
    assert near.score > hopeless.score


# --------------------------------------------------------------------------- #
# The evaluator is total: nothing a candidate does can raise
# --------------------------------------------------------------------------- #


def test_verdict_is_the_frozen_schema_type() -> None:
    task = next(t for t in SUITE.tasks if t.task_id == "math-train-metres")
    verdict = EVALUATE(task, _trajectory("math-train-metres", "18000"))
    assert isinstance(verdict, EvaluatorVerdict)
    assert verdict.evaluator_id == "code_math-exec-v1"
    assert verdict.passed and verdict.score == 1.0


def test_missing_final_answer_fails_cleanly() -> None:
    task = next(t for t in SUITE.tasks if t.task_id == "math-train-metres")
    verdict = EVALUATE(task, _trajectory("math-train-metres", None))
    assert not verdict.passed and verdict.score == 0.0
    assert "no final answer" in verdict.rationale


@pytest.mark.parametrize(
    "answer",
    ["", "   ", "I am not sure", "def count_multiples(:\n", "print('hello')"],
    ids=["empty", "blank", "prose", "syntax-error", "no-entry-point"],
)
def test_junk_code_submissions_score_zero(answer: str) -> None:
    task = next(t for t in SUITE.tasks if t.task_id == "code-inclusive-multiples")
    verdict = EVALUATE(task, _trajectory("code-inclusive-multiples", answer))
    assert not verdict.passed and verdict.score == 0.0


def test_answer_wrapped_in_a_code_fence_is_accepted() -> None:
    task = next(t for t in SUITE.tasks if t.task_id == "code-inclusive-multiples")
    fenced = "```python\ndef count_multiples(lo, hi, k):\n    return hi // k - (lo - 1) // k\n```"
    verdict = EVALUATE(task, _trajectory("code-inclusive-multiples", fenced))
    assert verdict.passed


@pytest.mark.parametrize(
    "answer", ["18000", " 18000 ", "18,000", "The answer is 18000 metres."]
)
def test_math_answer_packaging_is_tolerated(answer: str) -> None:
    """Lenient about formatting, strict about value."""
    task = next(t for t in SUITE.tasks if t.task_id == "math-train-metres")
    assert EVALUATE(task, _trajectory("math-train-metres", answer)).passed


@pytest.mark.parametrize("answer", ["18001", "18", "19000", "no idea"])
def test_math_wrong_values_are_rejected(answer: str) -> None:
    task = next(t for t in SUITE.tasks if t.task_id == "math-train-metres")
    assert not EVALUATE(task, _trajectory("math-train-metres", answer)).passed


def test_a_hanging_submission_fails_instead_of_stalling() -> None:
    """The whole point of the timeout: a hang costs one budget, not the loop."""
    task = next(t for t in SUITE.tasks if t.task_id == "code-inclusive-multiples")
    verdict = evaluate_answer(
        task,
        "def count_multiples(lo, hi, k):\n    while True:\n        pass\n",
        timeout_seconds=1.0,
    )
    assert not verdict.passed and verdict.score == 0.0
    assert "timed out" in verdict.rationale


def test_a_submission_that_exits_the_process_fails_cleanly() -> None:
    task = next(t for t in SUITE.tasks if t.task_id == "code-inclusive-multiples")
    verdict = evaluate_answer(
        task, "import sys\ndef count_multiples(lo, hi, k):\n    return 0\nsys.exit(3)\n"
    )
    assert not verdict.passed and verdict.score == 0.0


def test_evaluation_is_deterministic() -> None:
    task = next(t for t in SUITE.tasks if t.task_id == "code-min-coins")
    submission = _TRAP_ANSWERS["code-min-coins"]
    scores = {evaluate_answer(task, submission).score for _ in range(3)}
    assert len(scores) == 1


def test_a_raised_runner_still_ends_up_in_the_denominator() -> None:
    """Mirrors the harness rule: a raising runner must not let the domain quietly disappear."""

    def flaky_runner(spec: AgentSpec, task: TaskSpec, attempt: int) -> Trajectory:
        if task.task_id == "code-median":
            raise RuntimeError("boom")
        return _runner(spec, task, attempt)

    evaluator = Evaluator([SUITE], evaluators={DOMAIN: EVALUATE}, repeats=3)
    report = evaluator.run_iteration(0, SPEC, flaky_runner)
    domain_report = report.domain(DOMAIN)
    assert domain_report.accuracy.sample_size == len(SUITE.tasks) * 3


# --------------------------------------------------------------------------- #
# The sandbox
# --------------------------------------------------------------------------- #


def test_sandbox_runs_ordinary_code() -> None:
    result = run_python("print(6 * 7)")
    assert result.ok and result.stdout.strip() == "42"


def test_sandbox_kills_an_infinite_loop() -> None:
    result = run_python("while True:\n    pass\n", timeout_seconds=1.0)
    assert result.timed_out and result.status == "timeout"


@pytest.mark.parametrize(
    "source",
    [
        "import socket; socket.socket()",
        "import socket; socket.create_connection(('example.com', 80))",
        "import subprocess; subprocess.run(['echo', 'hi'])",
        "import os; os.system('echo hi')",
    ],
    ids=["socket", "connect", "subprocess", "os-system"],
)
def test_sandbox_blocks_network_and_process_spawning(source: str) -> None:
    result = run_python(source)
    assert not result.ok
    assert "sandbox:" in result.stderr


def test_sandbox_survives_an_output_flood() -> None:
    result = run_python("for _ in range(200000):\n    print('x' * 50)\n", timeout_seconds=20.0)
    assert len(result.stdout) < 40_000
