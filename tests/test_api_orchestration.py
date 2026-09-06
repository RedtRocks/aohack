"""Tests for the api_orchestration domain: its task suite and its evaluator.

The load-bearing one is :func:`test_baseline_engine_scores_above_zero`. It
drives the domain through nothing but the public interface --
``agent_engineer.evaluation.Evaluator``, the domain's ``get_suite()`` and
``get_evaluator()``, and a ``TaskRunner`` that itself uses only
``TaskSpec.prompt`` and ``TaskSpec.metadata`` -- and asserts the resulting
accuracy is neither zero (no gradient for the engine to climb) nor one
(nothing left to improve).
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import pytest

from agent_engineer.domains.api_orchestration import get_evaluator, get_suite
from agent_engineer.domains.api_orchestration.api import ToolError
from agent_engineer.domains.api_orchestration.evaluator import RUBRICS, RUBRICS_BY_TASK, Hop
from agent_engineer.evaluation import Evaluator, TaskSpec
from agent_engineer.schemas import AgentSpec, EvaluatorVerdict, ToolCall, Trajectory

SUITE = get_suite()
EVALUATOR = get_evaluator()

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_SCALAR = (str, int, float, bool)

BASELINE_ROUNDS = 2

_SPEC = AgentSpec(spec_id="spec_baseline", system_prompt="You are a support operations agent.")

# --------------------------------------------------------------------------- #
# A domain-blind driver, standing in for the engine.
#
# It touches exactly the interface a real engine is allowed to touch: task
# prompts, the tool catalogue and dispatcher carried in TaskSpec.metadata, and
# the frozen Trajectory schema. It knows nothing about customers, orders or
# shipments.
#
# Its policy is deliberately shallow -- pull identifier-shaped tokens out of
# the prompt, call whatever tool has its required arguments satisfied, harvest
# the results by field name, repeat twice, then report everything it saw. That
# is roughly what an unoptimized agent does: it threads state one or two hops
# and then runs out of plan. Deep chains, recovery after an error, filtered
# retries and arithmetic are all beyond it, which is the gradient the loop is
# meant to close.
# --------------------------------------------------------------------------- #


def _harvest(value: Any, state: dict[str, Any], depth: int = 0) -> None:
    """Fold scalar fields of a tool result into the working state, first value wins."""
    if depth > 2:
        return
    if isinstance(value, dict):
        for name, item in value.items():
            if isinstance(item, _SCALAR) and item is not None and name not in state:
                state[name] = item
            elif isinstance(item, (dict, list)):
                _harvest(item, state, depth + 1)
    elif isinstance(value, list):
        for item in value[:1]:
            _harvest(item, state, depth + 1)


def baseline_runner(spec: AgentSpec, task: TaskSpec, attempt: int) -> Trajectory:
    """A shallow, task-metadata-only driver. Reproducible: `attempt` does not change it."""
    schemas = task.metadata["tool_schemas"]
    call_tool = task.metadata["call_tool"]

    state: dict[str, Any] = {}
    match = _EMAIL.search(task.prompt)
    if match:
        state["email"] = match.group(0)

    calls: list[ToolCall] = []
    seen: set[str] = set()
    observations: list[dict[str, Any]] = []
    started = time.monotonic()

    for _ in range(BASELINE_ROUNDS):
        for schema in schemas:
            required = schema["parameters"]["required"]
            if not required or any(name not in state for name in required):
                continue
            args = {name: state[name] for name in required}
            signature = f"{schema['name']}::{json.dumps(args, sort_keys=True, default=str)}"
            if signature in seen:
                continue
            seen.add(signature)
            try:
                result = call_tool(schema["name"], args)
            except ToolError as exc:
                calls.append(
                    ToolCall(ordinal=len(calls), tool_name=schema["name"], args=args, error=str(exc))
                )
                observations.append({"tool": schema["name"], "error": str(exc)})
                continue
            calls.append(
                ToolCall(ordinal=len(calls), tool_name=schema["name"], args=args, result=result)
            )
            observations.append({"tool": schema["name"], "result": result})
            _harvest(result, state)

    return Trajectory(
        trajectory_id=f"traj_baseline_{task.task_id}_{attempt}",
        spec_id=spec.spec_id,
        task_id=task.task_id,
        tool_calls=tuple(calls),
        final_answer=json.dumps(observations, sort_keys=True, default=str),
        elapsed_seconds=time.monotonic() - started,
    )


def reference_runner(spec: AgentSpec, task: TaskSpec, attempt: int) -> Trajectory:
    """Walk a rubric's hops for real and answer with its reference write-up.

    Not an agent: a proof that each rubric describes a chain the simulated API
    actually supports and an answer the graders actually accept.
    """
    rubric = RUBRICS_BY_TASK[task.task_id]
    call_tool = task.metadata["call_tool"]
    started = time.monotonic()
    calls: list[ToolCall] = []
    for hop in rubric.hops:
        args = dict(hop.args)
        try:
            result = call_tool(hop.tool, args)
        except ToolError as exc:
            calls.append(ToolCall(ordinal=len(calls), tool_name=hop.tool, args=args, error=str(exc)))
            continue
        calls.append(ToolCall(ordinal=len(calls), tool_name=hop.tool, args=args, result=result))
    return Trajectory(
        trajectory_id=f"traj_reference_{task.task_id}_{attempt}",
        spec_id=spec.spec_id,
        task_id=task.task_id,
        tool_calls=tuple(calls),
        final_answer=rubric.reference_answer,
        elapsed_seconds=time.monotonic() - started,
    )


# --------------------------------------------------------------------------- #
# The definition of done.
# --------------------------------------------------------------------------- #


def test_baseline_engine_scores_above_zero() -> None:
    """The engine's own Evaluator, given only this domain's suite and evaluator,
    produces a non-zero baseline accuracy -- and less than a perfect one.

    Zero would mean the loop cannot tell a bad agent from a worse one; one would
    mean there is nothing left for it to optimize.
    """
    harness = Evaluator(
        suites=(SUITE,), evaluators={"api_orchestration": EVALUATOR}, repeats=3
    )
    report = harness.run_iteration(0, _SPEC, baseline_runner)

    domain_report = report.domain("api_orchestration")
    accuracy = domain_report.accuracy.require()
    assert accuracy > 0.0, "baseline scored zero: this domain gives the engine no gradient"
    assert accuracy < 1.0, "baseline scored perfect: this domain gives the engine nothing to fix"


def test_reference_walk_scores_perfectly_through_the_real_harness() -> None:
    """The same Evaluator, driven by a runner that actually threads every value, tops out."""
    harness = Evaluator(
        suites=(SUITE,), evaluators={"api_orchestration": EVALUATOR}, repeats=3
    )
    report = harness.run_iteration(0, _SPEC, reference_runner)
    assert report.domain("api_orchestration").accuracy.require() == pytest.approx(1.0)


def test_runner_uses_only_the_public_interface() -> None:
    """The driver above reaches the domain through TaskSpec.prompt and .metadata only."""
    task = SUITE.tasks[0]
    trajectory = baseline_runner(_SPEC, task, 0)
    assert trajectory.tool_calls, "the driver found no usable tool schema in task.metadata"
    assert trajectory.tool_calls[0].tool_name == "find_customer"


def test_baseline_leaves_headroom_on_hard_tasks() -> None:
    """Shallow threading clears short chains and stalls on long ones."""
    scores: dict[str, float] = {}
    for task in SUITE.tasks:
        trajectory = baseline_runner(_SPEC, task, 0)
        scores[task.task_id] = EVALUATOR(task, trajectory).score

    difficulty = {task.task_id: task.metadata["difficulty"] for task in SUITE.tasks}
    easy = [scores[task_id] for task_id, level in difficulty.items() if level == "easy"]
    hard = [scores[task_id] for task_id, level in difficulty.items() if level in {"hard", "expert"}]
    assert (sum(easy) / len(easy)) > (sum(hard) / len(hard)), (
        "difficulty labels do not track what the shallow driver can actually do"
    )


# --------------------------------------------------------------------------- #
# Suite shape.
# --------------------------------------------------------------------------- #


def test_suite_domain_matches_every_task() -> None:
    assert SUITE.domain == "api_orchestration"
    assert all(task.domain == "api_orchestration" for task in SUITE.tasks)


def test_every_task_has_a_rubric_and_vice_versa() -> None:
    assert {task.task_id for task in SUITE.tasks} == set(RUBRICS_BY_TASK)


def test_tasks_are_multi_hop() -> None:
    """No task is a single-shot lookup except the one that exists to fail on the first call."""
    for task in SUITE.tasks:
        rubric = RUBRICS_BY_TASK[task.task_id]
        if task.task_id == "t_unknown_customer":
            assert len(rubric.hops) == 1
            continue
        assert len(rubric.hops) >= 2, f"{task.task_id} does not require threading a value"


def test_no_task_prompt_leaks_an_internal_identifier() -> None:
    """If a prompt already contains the id, the hop that fetches it proves nothing."""
    internal = re.compile(r"\b(c_|o_|s_|w_|sku_)\w+")
    for task in SUITE.tasks:
        leaked = internal.findall(task.prompt)
        assert not leaked, f"{task.task_id} leaks internal identifiers: {leaked}"


def test_difficulty_labels_range() -> None:
    levels = {task.metadata["difficulty"] for task in SUITE.tasks}
    assert levels <= {"easy", "medium", "hard", "expert"}
    assert len(levels) >= 3, "the set does not range"


def test_metadata_is_the_only_fixture_channel() -> None:
    """Everything a runner needs to act on a task lives in metadata, not on extra attributes."""
    task = SUITE.tasks[0]
    assert "tool_schemas" in task.metadata
    assert "call_tool" in task.metadata
    assert callable(task.metadata["call_tool"])


# --------------------------------------------------------------------------- #
# The domain's three shapes of hop actually appear.
# --------------------------------------------------------------------------- #


def _call_tool(task_id: str, tool_name: str, args: dict[str, Any]) -> Any:
    task = next(task for task in SUITE.tasks if task.task_id == task_id)
    return task.metadata["call_tool"](tool_name, args)


def test_a_result_narrows_which_call_comes_next() -> None:
    """A null shipment_id redirects the chain from tracking to inventory."""
    backordered = _call_tool("t_backorder_restock", "get_order", {"order_id": "o_1003"})
    assert backordered["shipment_id"] is None
    with pytest.raises(ToolError):
        _call_tool("t_backorder_restock", "track_shipment", {"shipment_id": "none"})
    inventory = _call_tool(
        "t_backorder_restock", "get_inventory", {"sku": "sku_gpu", "warehouse_id": "w_us2"}
    )
    assert inventory["restock_days"] == 14


def test_a_failed_call_names_its_own_recovery() -> None:
    with pytest.raises(ToolError) as excinfo:
        _call_tool("t_carrier_outage", "track_shipment", {"shipment_id": "s_503"})
    assert "list_shipment_events" in str(excinfo.value)
    events = _call_tool("t_carrier_outage", "list_shipment_events", {"shipment_id": "s_503"})
    assert events[-1] == {"when": "d-1", "state": "in_transit", "city": "Denver"}


def test_an_empty_result_is_a_result_not_an_error() -> None:
    assert _call_tool("t_no_orders", "list_orders", {"customer_id": "c_linus"}) == []
    filtered = _call_tool(
        "t_widen_after_empty", "list_orders", {"customer_id": "c_alan", "status": "shipped"}
    )
    assert filtered == []
    widened = _call_tool("t_widen_after_empty", "list_orders", {"customer_id": "c_alan"})
    assert [order["order_id"] for order in widened] == ["o_1005"]


def test_refund_refuses_an_undelivered_order() -> None:
    with pytest.raises(ToolError, match="not delivered"):
        _call_tool(
            "t_refund_not_delivered",
            "compute_refund",
            {"order_id": "o_1004", "policy_window_days": 60},
        )


# --------------------------------------------------------------------------- #
# The simulated API is deterministic and offline.
# --------------------------------------------------------------------------- #


def test_repeated_calls_are_identical() -> None:
    for _ in range(3):
        assert _call_tool("t_track_simple", "find_customer", {"email": "ada@lovelace.io"}) == {
            "customer_id": "c_ada",
            "name": "Ada Lovelace",
            "tier": "gold",
            "region": "EU",
        }


def test_baseline_run_is_reproducible() -> None:
    task = next(task for task in SUITE.tasks if task.task_id == "t_track_simple")
    first = baseline_runner(_SPEC, task, 0)
    second = baseline_runner(_SPEC, task, 1)
    assert first.tool_calls == second.tool_calls
    assert first.final_answer == second.final_answer


def test_unavailable_tool_is_refused() -> None:
    with pytest.raises(ToolError, match="not available"):
        _call_tool("t_no_orders", "definitely_not_a_tool", {})


def test_unknown_task_id_has_no_tasks_metadata() -> None:
    assert not any(task.task_id == "t_does_not_exist" for task in SUITE.tasks)


# --------------------------------------------------------------------------- #
# Evaluator behaviour.
# --------------------------------------------------------------------------- #


def _task(task_id: str) -> TaskSpec:
    return next(task for task in SUITE.tasks if task.task_id == task_id)


def _bare_trajectory(task_id: str, answer: str | None) -> Trajectory:
    return Trajectory(
        trajectory_id=f"traj_{task_id}", spec_id="spec_x", task_id=task_id, final_answer=answer
    )


def test_no_calls_and_no_answer_scores_zero() -> None:
    verdict = EVALUATOR(_task("t_track_simple"), _bare_trajectory("t_track_simple", None))
    assert verdict.score == 0.0
    assert not verdict.passed
    assert "no final answer" in verdict.rationale


def test_a_guessed_answer_without_the_chain_cannot_pass() -> None:
    """The right words with none of the calls is the failure mode this domain is built to catch."""
    verdict = EVALUATOR(
        _task("t_track_simple"),
        _bare_trajectory("t_track_simple", "Order o_1001 was last scanned in Rotterdam."),
    )
    assert not verdict.passed
    assert verdict.score == pytest.approx(0.7)
    assert "dependent calls never made" in verdict.rationale


def test_the_chain_without_an_answer_still_earns_partial_credit() -> None:
    trajectory = reference_runner(_SPEC, _task("t_track_simple"), 0)
    silent = trajectory.model_copy(update={"final_answer": None})
    verdict = EVALUATOR(_task("t_track_simple"), silent)
    assert verdict.score == pytest.approx(0.3)
    assert not verdict.passed


def test_a_forbidden_claim_zeroes_the_answer_half() -> None:
    trajectory = reference_runner(_SPEC, _task("t_unknown_customer"), 0)
    wrong = trajectory.model_copy(update={"final_answer": "That account is gold tier."})
    verdict = EVALUATOR(_task("t_unknown_customer"), wrong)
    assert verdict.score == pytest.approx(0.3)
    assert "contradicted" in verdict.rationale


def test_hop_matching_requires_the_threaded_argument() -> None:
    """A hop is matched by the value carried in, not by the tool name."""
    hop = Hop("get_order", {"order_id": "o_1002"})
    right = Trajectory(
        trajectory_id="t1", spec_id="s", task_id="t_refund_amount",
        tool_calls=(ToolCall(ordinal=0, tool_name="get_order", args={"order_id": "o_1002"}, result={}),),
    )
    wrong = Trajectory(
        trajectory_id="t2", spec_id="s", task_id="t_refund_amount",
        tool_calls=(ToolCall(ordinal=0, tool_name="get_order", args={"order_id": "o_1001"}, result={}),),
    )
    assert hop.matched_by(right)
    assert not hop.matched_by(wrong)


def test_forbidden_args_distinguish_the_widened_retry() -> None:
    hop = Hop("list_orders", {"customer_id": "c_alan"}, forbidden_args=("status",))
    filtered = Trajectory(
        trajectory_id="t3", spec_id="s", task_id="t_widen_after_empty",
        tool_calls=(
            ToolCall(
                ordinal=0, tool_name="list_orders",
                args={"customer_id": "c_alan", "status": "shipped"}, result=[],
            ),
        ),
    )
    assert not hop.matched_by(filtered)


def test_an_error_hop_needs_the_call_to_have_failed() -> None:
    hop = Hop("find_customer", {"email": "nobody@nowhere.example"}, status="error")
    succeeded = Trajectory(
        trajectory_id="t4", spec_id="s", task_id="t_unknown_customer",
        tool_calls=(
            ToolCall(
                ordinal=0, tool_name="find_customer",
                args={"email": "nobody@nowhere.example"}, result={"tier": "gold"},
            ),
        ),
    )
    assert not hop.matched_by(succeeded)


def test_answers_are_matched_past_formatting() -> None:
    trajectory = reference_runner(_SPEC, _task("t_open_order_value"), 0)
    formatted = trajectory.model_copy(
        update={"final_answer": "Ada's non-cancelled orders total **$33,400**."}
    )
    assert EVALUATOR(_task("t_open_order_value"), formatted).passed


def test_evaluator_refuses_a_task_it_does_not_grade() -> None:
    with pytest.raises(KeyError):
        EVALUATOR(_bare_task("t_from_another_domain"), _bare_trajectory("t_from_another_domain", "hello"))


def _bare_task(task_id: str) -> TaskSpec:
    return TaskSpec(task_id=task_id, domain="somewhere_else", prompt="n/a")


def test_verdict_carries_this_evaluators_id() -> None:
    verdict = EVALUATOR(_task("t_refund_window"), reference_runner(_SPEC, _task("t_refund_window"), 0))
    assert verdict.evaluator_id == "api_orchestration.dependency_evaluator.v1"
    assert isinstance(verdict, EvaluatorVerdict)


def test_evaluator_never_aggregates() -> None:
    """The evaluator returns a single verdict; it has no notion of a whole suite."""
    assert not hasattr(EVALUATOR, "run_iteration")
    assert not hasattr(EVALUATOR, "accuracy")


# --------------------------------------------------------------------------- #
# The dependency runs one way: this domain never imports the engine.
# --------------------------------------------------------------------------- #


def test_this_domain_imports_no_engine_module() -> None:
    """Only the frozen schemas and the measurement interface are fair game."""
    import pathlib

    package = pathlib.Path(__file__).resolve().parent.parent / "agent_engineer" / "domains" / "api_orchestration"
    allowed_prefixes = ("agent_engineer.schemas", "agent_engineer.evaluation", "agent_engineer.domains.api_orchestration")
    for path in package.glob("*.py"):
        for match in re.finditer(r"^\s*(?:from|import)\s+(agent_engineer[\w.]*)", path.read_text(), re.M):
            module = match.group(1)
            assert module.startswith(allowed_prefixes), f"{path.name} imports {module}"
