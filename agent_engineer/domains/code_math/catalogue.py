"""The code_math task catalogue: sixteen problems with ground truth attached.

Every task is settled by execution or exact comparison, never by a judge, so the
same submission scores the same today and next week. Two shapes share the
:class:`~agent_engineer.evaluation.TaskSpec` container, distinguished by
``metadata["kind"]``:

* **code** -- the agent writes a Python function; hidden test cases run it in a
  sandboxed subprocess (see :mod:`.sandbox`). ``metadata["hidden_cases"]`` holds
  the graded cases, ``metadata["entry_point"]`` the function name the
  submission must define, and ``TaskSpec.expected`` a known-good reference
  implementation -- present so the set is self-checking, never used to grade.
* **math** -- the agent produces a value; ``TaskSpec.expected`` is the ground
  truth and ``metadata["tolerance"]`` the absolute tolerance for the numeric
  comparison (0.0 means exact).

Every task also carries ``metadata["trap"]``: the specific near-miss a
plausible-but-sloppy attempt produces (an off-by-one, a unit slip, the right
method on the wrong edge case). It documents intent for humans reading the set
and is never shown to the agent; the hidden cases are chosen so that near-miss
actually fails. Difficulty spans easy to hard on purpose -- an all-pass set
gives the engine nothing to improve, an all-fail set gives it no gradient.
"""

from __future__ import annotations

from agent_engineer.evaluation import DomainSuite, TaskSpec

__all__ = ["DOMAIN", "load_suite"]

DOMAIN = "code_math"


def _code_task(
    task_id: str,
    *,
    difficulty: str,
    entry_point: str,
    prompt: str,
    trap: str,
    hidden_cases: tuple[dict[str, str], ...],
    reference_solution: str,
) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        domain=DOMAIN,
        prompt=prompt,
        expected=reference_solution,
        metadata={
            "kind": "code",
            "difficulty": difficulty,
            "trap": trap,
            "entry_point": entry_point,
            "hidden_cases": list(hidden_cases),
        },
    )


def _math_task(
    task_id: str,
    *,
    difficulty: str,
    prompt: str,
    trap: str,
    expected: str,
    tolerance: float = 0.0,
) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        domain=DOMAIN,
        prompt=prompt,
        expected=expected,
        metadata={
            "kind": "math",
            "difficulty": difficulty,
            "trap": trap,
            "tolerance": tolerance,
        },
    )


_CODE_TASKS: tuple[TaskSpec, ...] = (
    _code_task(
        "code-inclusive-multiples",
        difficulty="easy",
        entry_point="count_multiples",
        prompt=(
            "Write a Python function count_multiples(lo, hi, k) that returns how many "
            "integers n with lo <= n <= hi are multiples of k. BOTH bounds are inclusive, "
            "lo <= hi, k is a positive integer, and lo and hi may be negative or zero. "
            "Example: count_multiples(1, 10, 3) == 3.\n\n"
            "Reply with the function definition as Python source and nothing else."
        ),
        trap="Exclusive upper bound, or missing that 0 and negative numbers are multiples too.",
        hidden_cases=(
            {"call": "count_multiples(1, 10, 3)", "expected": "3"},
            {"call": "count_multiples(0, 0, 5)", "expected": "1"},
            {"call": "count_multiples(-6, 6, 3)", "expected": "5"},
            {"call": "count_multiples(7, 7, 7)", "expected": "1"},
            {"call": "count_multiples(1, 10, 11)", "expected": "0"},
        ),
        reference_solution="def count_multiples(lo, hi, k):\n    return hi // k - (lo - 1) // k\n",
    ),
    _code_task(
        "code-median",
        difficulty="easy",
        entry_point="median",
        prompt=(
            "Write a Python function median(nums) that returns the median of a non-empty "
            "list of numbers as a float. The input is not necessarily sorted and must not be "
            "mutated. For an even-length list return the mean of the two middle values. "
            "Example: median([1, 2, 3, 4]) == 2.5.\n\n"
            "Reply with the function definition as Python source and nothing else."
        ),
        trap="Assuming sorted input, or taking the lower middle element on even lengths.",
        hidden_cases=(
            {"call": "median([3, 1, 2])", "expected": "2.0"},
            {"call": "median([1, 2, 3, 4])", "expected": "2.5"},
            {"call": "median([5])", "expected": "5.0"},
            {"call": "median([2, 1])", "expected": "1.5"},
            {"call": "median([7, 1, 3, 2])", "expected": "2.5"},
        ),
        reference_solution=(
            "def median(nums):\n"
            "    ordered = sorted(nums)\n"
            "    mid = len(ordered) // 2\n"
            "    if len(ordered) % 2:\n"
            "        return float(ordered[mid])\n"
            "    return (ordered[mid - 1] + ordered[mid]) / 2\n"
        ),
    ),
    _code_task(
        "code-roman-to-int",
        difficulty="medium",
        entry_point="roman_to_int",
        prompt=(
            "Write a Python function roman_to_int(s) that converts a valid uppercase Roman "
            "numeral between I and MMMCMXCIX into an int. Handle the subtractive forms IV, "
            "IX, XL, XC, CD and CM. Example: roman_to_int('LVIII') == 58.\n\n"
            "Reply with the function definition as Python source and nothing else."
        ),
        trap="Summing letter values without subtractive pairs: MCMXCIV comes out as 2216.",
        hidden_cases=(
            {"call": "roman_to_int('LVIII')", "expected": "58"},
            {"call": "roman_to_int('IX')", "expected": "9"},
            {"call": "roman_to_int('MCMXCIV')", "expected": "1994"},
            {"call": "roman_to_int('MMMCMXCIX')", "expected": "3999"},
            {"call": "roman_to_int('IV')", "expected": "4"},
        ),
        reference_solution=(
            "def roman_to_int(s):\n"
            "    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n"
            "    total = 0\n"
            "    for index, char in enumerate(s):\n"
            "        value = values[char]\n"
            "        if index + 1 < len(s) and value < values[s[index + 1]]:\n"
            "            total -= value\n"
            "        else:\n"
            "            total += value\n"
            "    return total\n"
        ),
    ),
    _code_task(
        "code-merge-intervals",
        difficulty="medium",
        entry_point="merge_intervals",
        prompt=(
            "Write a Python function merge_intervals(intervals) taking a list of CLOSED "
            "intervals [start, end] and returning the merged, non-overlapping intervals as a "
            "list of [start, end] lists sorted by start. The input may arrive in any order "
            "and may be empty. Because the intervals are closed, ones that merely touch, such "
            "as [1, 4] and [4, 5], do overlap and merge into [1, 5].\n\n"
            "Reply with the function definition as Python source and nothing else."
        ),
        trap="Treating touching intervals as disjoint, or assuming the input arrives sorted.",
        hidden_cases=(
            {
                "call": "merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]])",
                "expected": "[[1, 6], [8, 10], [15, 18]]",
            },
            {"call": "merge_intervals([[1, 4], [4, 5]])", "expected": "[[1, 5]]"},
            {"call": "merge_intervals([])", "expected": "[]"},
            {"call": "merge_intervals([[5, 6], [1, 2]])", "expected": "[[1, 2], [5, 6]]"},
            {"call": "merge_intervals([[1, 10], [2, 3]])", "expected": "[[1, 10]]"},
        ),
        reference_solution=(
            "def merge_intervals(intervals):\n"
            "    merged = []\n"
            "    for start, end in sorted(intervals):\n"
            "        if merged and start <= merged[-1][1]:\n"
            "            merged[-1][1] = max(merged[-1][1], end)\n"
            "        else:\n"
            "            merged.append([start, end])\n"
            "    return merged\n"
        ),
    ),
    _code_task(
        "code-first-index",
        difficulty="medium",
        entry_point="first_index",
        prompt=(
            "Write a Python function first_index(arr, target) returning the index of the "
            "FIRST occurrence of target in the sorted (non-decreasing) list arr, or -1 if "
            "target is absent. The list may contain duplicates and may be empty, and the "
            "search must run in O(log n). Example: first_index([1, 2, 2, 2, 3], 2) == 1.\n\n"
            "Reply with the function definition as Python source and nothing else."
        ),
        trap="A textbook binary search returns whichever duplicate it lands on, not the first.",
        hidden_cases=(
            {"call": "first_index([1, 2, 2, 2, 3], 2)", "expected": "1"},
            {"call": "first_index([1, 2, 3], 4)", "expected": "-1"},
            {"call": "first_index([], 1)", "expected": "-1"},
            {"call": "first_index([2, 2], 2)", "expected": "0"},
            {"call": "first_index([1, 2, 3], 1)", "expected": "0"},
            {"call": "first_index([1, 2, 3], 3)", "expected": "2"},
        ),
        reference_solution=(
            "def first_index(arr, target):\n"
            "    lo, hi = 0, len(arr)\n"
            "    while lo < hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if arr[mid] < target:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid\n"
            "    if lo < len(arr) and arr[lo] == target:\n"
            "        return lo\n"
            "    return -1\n"
        ),
    ),
    _code_task(
        "code-top-words",
        difficulty="medium",
        entry_point="top_words",
        prompt=(
            "Write a Python function top_words(text, n) returning the n most frequent words "
            "in text as a list of (word, count) tuples. A word is a maximal run of ASCII "
            "letters, compared case-insensitively and reported in lower case; punctuation and "
            "digits separate words. Sort by count descending, breaking ties alphabetically. "
            "If text holds fewer than n distinct words, return them all.\n\n"
            "Reply with the function definition as Python source and nothing else."
        ),
        trap="Unstable tie-breaking, punctuation left attached, or original case preserved.",
        hidden_cases=(
            {"call": "top_words('the cat the hat', 2)", "expected": "[('the', 2), ('cat', 1)]"},
            {"call": "top_words('a b a b c', 2)", "expected": "[('a', 2), ('b', 2)]"},
            {"call": "top_words('Hello, hello world!', 1)", "expected": "[('hello', 2)]"},
            {"call": "top_words('x', 5)", "expected": "[('x', 1)]"},
            {"call": "top_words('', 3)", "expected": "[]"},
            {
                "call": "top_words('zebra apple zebra apple', 2)",
                "expected": "[('apple', 2), ('zebra', 2)]",
            },
        ),
        reference_solution=(
            "import re\n"
            "from collections import Counter\n"
            "\n"
            "def top_words(text, n):\n"
            "    counts = Counter(re.findall(r'[A-Za-z]+', text.lower()))\n"
            "    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))\n"
            "    return ranked[:n]\n"
        ),
    ),
    _code_task(
        "code-max-concurrent",
        difficulty="hard",
        entry_point="max_concurrent",
        prompt=(
            "Write a Python function max_concurrent(meetings) taking a list of HALF-OPEN "
            "intervals [start, end) and returning the largest number of meetings in progress "
            "at any single instant. Half-open means a meeting ending at time t and one "
            "starting at time t do NOT overlap, so max_concurrent([[1, 5], [5, 9]]) == 1. "
            "The list may be empty and is not necessarily sorted.\n\n"
            "Reply with the function definition as Python source and nothing else."
        ),
        trap="A closed-interval sweep orders starts before ends and reports 2 for [[1,5],[5,9]].",
        hidden_cases=(
            {"call": "max_concurrent([[0, 30], [5, 10], [15, 20]])", "expected": "2"},
            {"call": "max_concurrent([[1, 5], [5, 9]])", "expected": "1"},
            {"call": "max_concurrent([])", "expected": "0"},
            {"call": "max_concurrent([[1, 4], [2, 5], [3, 6]])", "expected": "3"},
            {"call": "max_concurrent([[0, 1], [0, 1], [0, 1]])", "expected": "3"},
        ),
        reference_solution=(
            "def max_concurrent(meetings):\n"
            "    events = []\n"
            "    for start, end in meetings:\n"
            "        events.append((start, 1))\n"
            "        events.append((end, -1))\n"
            "    events.sort(key=lambda event: (event[0], event[1]))\n"
            "    best = running = 0\n"
            "    for _, delta in events:\n"
            "        running += delta\n"
            "        if running > best:\n"
            "            best = running\n"
            "    return best\n"
        ),
    ),
    _code_task(
        "code-min-coins",
        difficulty="hard",
        entry_point="min_coins",
        prompt=(
            "Write a Python function min_coins(coins, amount) returning the fewest coins from "
            "the list coins (each denomination usable any number of times) that sum to exactly "
            "amount, or -1 when no combination does. amount is a non-negative int and "
            "min_coins(coins, 0) == 0 for any coins. A greedy largest-coin-first strategy is "
            "not correct here.\n\n"
            "Reply with the function definition as Python source and nothing else."
        ),
        trap="Greedy selection: on [186, 419, 83, 408] for 6249 it misses the 20-coin optimum.",
        hidden_cases=(
            {"call": "min_coins([1, 2, 5], 11)", "expected": "3"},
            {"call": "min_coins([2], 3)", "expected": "-1"},
            {"call": "min_coins([1], 0)", "expected": "0"},
            {"call": "min_coins([2, 5], 9)", "expected": "3"},
            {"call": "min_coins([186, 419, 83, 408], 6249)", "expected": "20"},
        ),
        reference_solution=(
            "def min_coins(coins, amount):\n"
            "    best = [0] + [float('inf')] * amount\n"
            "    for value in range(1, amount + 1):\n"
            "        for coin in coins:\n"
            "            if coin <= value and best[value - coin] + 1 < best[value]:\n"
            "                best[value] = best[value - coin] + 1\n"
            "    return -1 if best[amount] == float('inf') else best[amount]\n"
        ),
    ),
)

_MATH_TASKS: tuple[TaskSpec, ...] = (
    _math_task(
        "math-train-metres",
        difficulty="easy",
        prompt=(
            "A train travels at a constant 72 km/h. How many METRES does it cover in 15 "
            "minutes? Reply with the number alone, no units and no thousands separators."
        ),
        trap="Answering in kilometres (18), or multiplying 72 by 15 with no unit conversion.",
        expected="18000",
    ),
    _math_task(
        "math-divisible-3-or-5",
        difficulty="easy",
        prompt=(
            "Find the sum of all positive integers strictly below 1000 that are divisible by "
            "3 or by 5. Reply with the number alone."
        ),
        trap="Adding the multiples of 3 and of 5 separately, double-counting multiples of 15.",
        expected="233168",
    ),
    _math_task(
        "math-modular-power",
        difficulty="medium",
        prompt="Compute 7^222 mod 13. Reply with the number alone.",
        trap="Reducing the exponent mod 13 instead of mod 12, the order of the unit group.",
        expected="12",
    ),
    _math_task(
        "math-exactly-two-heads",
        difficulty="medium",
        prompt=(
            "A fair coin is flipped 5 times. What is the probability of getting exactly 2 "
            "heads? Reply with the exact decimal value alone."
        ),
        trap="Reporting 0.25 for two independent heads, ignoring the 10 orderings out of 32.",
        expected="0.3125",
        tolerance=1e-9,
    ),
    _math_task(
        "math-compound-quarterly",
        difficulty="medium",
        prompt=(
            "1200 is invested at a 5% nominal annual interest rate compounded QUARTERLY. What "
            "is the account value after 3 years, rounded to the nearest cent? Reply with the "
            "number alone, no currency symbol and no thousands separators."
        ),
        trap="Compounding annually (1389.15) or applying simple interest (1380).",
        expected="1392.91",
        tolerance=0.005,
    ),
    _math_task(
        "math-distinct-even-4digit",
        difficulty="hard",
        prompt=(
            "How many 4-digit positive integers have four distinct digits and are even? Reply "
            "with the number alone."
        ),
        trap="Forgetting the leading digit cannot be 0, or that a trailing 0 frees it up again.",
        expected="2296",
    ),
    _math_task(
        "math-average-speed",
        difficulty="hard",
        prompt=(
            "A car drives 60 km at a steady 30 km/h, then immediately returns along the same "
            "60 km at a steady 60 km/h. What is its average speed for the whole trip, in km/h, "
            "rounded to two decimal places? Reply with the number alone."
        ),
        trap="Averaging the two speeds to 45 instead of total distance over total time.",
        expected="40.00",
        tolerance=0.005,
    ),
    _math_task(
        "math-pool-path-width",
        difficulty="hard",
        prompt=(
            "A rectangular swimming pool measures 25 m by 10 m. It is surrounded by a path of "
            "uniform width w metres on all four sides. The area of the path equals the area of "
            "the pool. Find w in metres, rounded to three decimal places. Reply with the "
            "number alone."
        ),
        trap="Using (25 + w)(10 + w), which adds the width to only one side of each dimension.",
        expected="3.042",
        tolerance=0.0005,
    ),
)

_SUITE = DomainSuite(domain=DOMAIN, tasks=_CODE_TASKS + _MATH_TASKS)


def load_suite() -> DomainSuite:
    """Return the frozen code_math suite. Built once at import; cheap to call."""
    return _SUITE
