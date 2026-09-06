"""The evaluator for the multi-step API orchestration domain.

Grading has two halves, because a domain about dependency has to score two
different things:

* **The answer.** Did the final text contain the facts only the full chain
  could have produced, and did it avoid the claims that mark a guess?
* **The chain.** Were the dependent calls actually made, with the *threaded*
  identifiers as arguments? A rubric hop pins ``get_order(order_id="o_1002")``,
  and ``o_1002`` appears nowhere in the prompt -- so matching that hop is proof
  the agent carried a value forward rather than pattern-matching the question.

Splitting them is what gives the engineer loop a gradient. An agent that walks
the chain but garbles the write-up scores differently from one that fabricates
a plausible answer without calling anything, and those two failures want
different mutations.

A trajectory passes only when both halves are complete. Partial credit is real
but never passing.

This module grades one trajectory at a time and nothing more: no accuracy, no
cost, no aggregation. Per ``agent_engineer/evaluation/INTERFACE.md`` that
counting belongs to the harness. The engine never imports this module; it is
reached only through :func:`get_evaluator`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agent_engineer.evaluation import TaskSpec
from agent_engineer.schemas import EvaluatorVerdict, Trajectory

EVALUATOR_ID = "api_orchestration.dependency_evaluator.v1"

ANSWER_WEIGHT = 0.7
CHAIN_WEIGHT = 0.3

_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Fold text to the form patterns are matched against.

    Lowercased, thousands separators and currency symbols stripped, whitespace
    collapsed -- so ``"$1,299.00"``, ``"1299"`` and ``"1,299"`` compare the same
    way and an answer is not marked wrong over formatting.
    """
    lowered = text.lower().replace(",", "").replace("$", "").replace("*", "")
    return _WHITESPACE.sub(" ", lowered).strip()


@dataclass(frozen=True)
class Fact:
    """One thing the answer must contain, and the spellings that count as containing it."""

    label: str
    patterns: tuple[str, ...]

    def matched_by(self, answer: str) -> bool:
        return any(_normalize(pattern) in answer for pattern in self.patterns)


@dataclass(frozen=True)
class Hop:
    """One dependent call the chain must contain.

    ``args`` are matched as a subset, compared as normalized strings, so an
    agent is free to pass extra arguments or spell an id with different casing.
    ``forbidden_args`` names arguments that must be *absent* -- that is how the
    widen-after-empty task distinguishes the filtered call from the unfiltered
    retry. An argument passed explicitly as ``None`` counts as absent.
    """

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    forbidden_args: tuple[str, ...] = ()
    status: str = "ok"
    """ok, error, or any -- some hops are only meaningful because they failed."""

    def matched_by(self, trajectory: Trajectory) -> bool:
        return any(self._matches(call) for call in trajectory.tool_calls)

    def _matches(self, call: Any) -> bool:
        if call.tool_name != self.tool:
            return False
        if self.status == "ok" and not call.succeeded:
            return False
        if self.status == "error" and call.succeeded:
            return False
        for name in self.forbidden_args:
            if call.args.get(name) is not None:
                return False
        for name, expected in self.args.items():
            if name not in call.args:
                return False
            if _normalize(str(call.args[name])) != _normalize(str(expected)):
                return False
        return True


@dataclass(frozen=True)
class Rubric:
    """How one task is graded, plus a worked answer used only to prove it is solvable."""

    task_id: str
    facts: tuple[Fact, ...]
    hops: tuple[Hop, ...]
    forbidden: tuple[str, ...] = ()
    """Phrases that mark a wrong claim. Any hit zeroes the answer half."""
    reference_answer: str = ""
    """A correct write-up. Never shown to an agent; the test suite uses it to
    prove every rubric is satisfiable from real tool output."""


_ADA = "ada@lovelace.io"
_GRACE = "grace@hopper.mil"
_ALAN = "alan@turing.uk"
_KATHERINE = "katherine@johnson.nasa"

RUBRICS: tuple[Rubric, ...] = (
    Rubric(
        task_id="t_refund_window",
        facts=(
            Fact("tier", ("gold",)),
            Fact("window_days", ("30",)),
        ),
        hops=(
            Hop("find_customer", {"email": _ADA}),
            Hop("get_refund_policy", {"tier": "gold", "region": "EU"}),
        ),
        forbidden=("platinum", "standard"),
        reference_answer="Ada Lovelace is a gold-tier customer in EU; her refund window is 30 days.",
    ),
    Rubric(
        task_id="t_last_scan",
        facts=(
            Fact("order_id", ("o_1005",)),
            Fact("city", ("manchester",)),
        ),
        hops=(
            Hop("find_customer", {"email": _ALAN}),
            Hop("list_orders", {"customer_id": "c_alan"}),
            Hop("get_order", {"order_id": "o_1005"}),
            Hop("track_shipment", {"shipment_id": "s_504"}),
        ),
        reference_answer="Order o_1005 was last scanned in Manchester, where it was delivered.",
    ),
    Rubric(
        task_id="t_track_simple",
        facts=(
            Fact("order_id", ("o_1001",)),
            Fact("city", ("rotterdam",)),
        ),
        hops=(
            Hop("find_customer", {"email": _ADA}),
            Hop("list_orders", {"customer_id": "c_ada"}),
            Hop("get_order", {"order_id": "o_1001"}),
            Hop("track_shipment", {"shipment_id": "s_501"}),
        ),
        forbidden=("delivered",),
        reference_answer=(
            "Order o_1001 is in transit; its last carrier scan was in Rotterdam, ETA 2 days."
        ),
    ),
    Rubric(
        task_id="t_backorder_restock",
        facts=(Fact("restock_days", ("14",)),),
        hops=(
            Hop("find_customer", {"email": _GRACE}),
            Hop("list_orders", {"customer_id": "c_grace"}),
            Hop("get_order", {"order_id": "o_1003"}),
            Hop("get_inventory", {"sku": "sku_gpu", "warehouse_id": "w_us2"}),
        ),
        reference_answer=(
            "Order o_1003 is backordered at warehouse w_us2; sku_gpu restocks there in 14 days."
        ),
    ),
    Rubric(
        task_id="t_carrier_outage",
        facts=(
            Fact("city", ("denver",)),
            Fact("state", ("in_transit", "in transit")),
        ),
        hops=(
            Hop("find_customer", {"email": _GRACE}),
            Hop("list_orders", {"customer_id": "c_grace"}),
            Hop("get_order", {"order_id": "o_1004"}),
            Hop("list_shipment_events", {"shipment_id": "s_503"}),
        ),
        forbidden=("delivered", "unable to determine", "could not determine"),
        reference_answer=(
            "Live tracking for FLEETWAY is down, so the stored scan log for s_503 was used: "
            "the parcel for order o_1004 is in_transit, last scanned in Denver."
        ),
    ),
    Rubric(
        task_id="t_no_orders",
        facts=(
            Fact(
                "none",
                ("no orders", "none", "zero orders", "0 orders", "does not have any", "no order"),
            ),
        ),
        hops=(
            Hop("find_customer", {"email": "linus@torvalds.fi"}),
            Hop("list_orders", {"customer_id": "c_linus"}),
        ),
        forbidden=("o_10",),
        reference_answer="Linus Torvalds has no orders on file: the account shows 0 orders.",
    ),
    Rubric(
        task_id="t_widen_after_empty",
        facts=(
            Fact(
                "no_shipped",
                ("no shipped", "none shipped", "no current shipped", "0 shipped", "does not have any shipped"),
            ),
            Fact("order_id", ("o_1005",)),
            Fact("status", ("delivered",)),
        ),
        hops=(
            Hop("find_customer", {"email": _ALAN}),
            Hop("list_orders", {"customer_id": "c_alan", "status": "shipped"}),
            Hop("list_orders", {"customer_id": "c_alan"}, forbidden_args=("status",)),
        ),
        reference_answer=(
            "Alan Turing has no shipped orders. The one order on his account is o_1005, "
            "which is delivered."
        ),
    ),
    Rubric(
        task_id="t_refund_amount",
        facts=(
            Fact("refund_cents", ("8900",)),
            Fact(
                "auto_approved",
                ("auto-approve", "auto_approve", "automatic approval", "automatically approved", "clears the"),
            ),
        ),
        hops=(
            Hop("find_customer", {"email": _ADA}),
            Hop("list_orders", {"customer_id": "c_ada"}),
            Hop("get_order", {"order_id": "o_1002"}),
            Hop("get_refund_policy", {"tier": "gold", "region": "EU"}),
            Hop("compute_refund", {"order_id": "o_1002", "policy_window_days": 30}),
        ),
        forbidden=("not auto", "does not clear", "below the automatic"),
        reference_answer=(
            "Order o_1002 was delivered 12 days ago, inside the 30-day gold/EU window, so the "
            "refund is 8900 cents. That is under the 10000-cent threshold, so it clears the "
            "automatic approval limit."
        ),
    ),
    Rubric(
        task_id="t_refund_not_delivered",
        facts=(
            Fact("verdict_no", ("no", "cannot", "can not", "not eligible")),
            Fact(
                "reason",
                ("not delivered", "has not been delivered", "not yet delivered", "still shipped"),
            ),
        ),
        hops=(
            Hop("find_customer", {"email": _GRACE}),
            Hop("list_orders", {"customer_id": "c_grace"}),
            Hop("get_order", {"order_id": "o_1004"}),
            Hop("compute_refund", {"order_id": "o_1004"}, status="any"),
        ),
        forbidden=("yes,", "yes -", "yes she", "refund of 4500"),
        reference_answer=(
            "No. Order o_1004 is shipped and not delivered yet, and compute_refund rejects it "
            "for exactly that reason, so no refund can be issued today."
        ),
    ),
    Rubric(
        task_id="t_soonest_warehouse",
        facts=(
            Fact("warehouse", ("w_us3",)),
            Fact(
                "soonest",
                ("immediat", "in stock", "on hand", "0 days", "today", "right away", "no wait", "already"),
            ),
        ),
        hops=(
            Hop("find_customer", {"email": _KATHERINE}),
            Hop("list_orders", {"customer_id": "c_katherine"}),
            Hop("get_order", {"order_id": "o_1006"}),
            Hop("list_warehouses", {"region": "US"}),
            Hop("get_inventory", {"sku": "sku_gpu", "warehouse_id": "w_us1"}),
            Hop("get_inventory", {"sku": "sku_gpu", "warehouse_id": "w_us2"}),
            Hop("get_inventory", {"sku": "sku_gpu", "warehouse_id": "w_us3"}),
        ),
        reference_answer=(
            "Order o_1006 needs 2 x sku_gpu. Of the US warehouses, w_us1 restocks in 5 days and "
            "w_us2 in 14, but w_us3 has 3 on hand, so w_us3 can ship it immediately."
        ),
    ),
    Rubric(
        task_id="t_unknown_customer",
        facts=(
            Fact(
                "not_found",
                ("no customer", "not found", "does not exist", "no account", "no such", "unknown"),
            ),
        ),
        hops=(Hop("find_customer", {"email": "nobody@nowhere.example"}, status="error"),),
        forbidden=("gold", "platinum", "standard"),
        reference_answer=(
            "There is no customer found for nobody@nowhere.example, so there is no tier to "
            "report."
        ),
    ),
    Rubric(
        task_id="t_open_order_value",
        facts=(Fact("total_cents", ("33400",)),),
        hops=(
            Hop("find_customer", {"email": _ADA}),
            Hop("list_orders", {"customer_id": "c_ada"}, forbidden_args=("status",)),
        ),
        forbidden=("38400",),
        reference_answer=(
            "Ada's non-cancelled orders are o_1001 at 24500 and o_1002 at 8900, so 33400 cents "
            "in total. o_1007 is cancelled and excluded."
        ),
    ),
)

RUBRICS_BY_TASK: dict[str, Rubric] = {rubric.task_id: rubric for rubric in RUBRICS}


class DependencyEvaluator:
    """Grades a trajectory on answer facts and on the dependent calls behind them.

    Matches the ``TaskEvaluator`` protocol: ``(task, trajectory) -> EvaluatorVerdict``.
    Grades one trajectory at a time; never aggregates.
    """

    def __init__(
        self,
        evaluator_id: str = EVALUATOR_ID,
        rubrics: dict[str, Rubric] | None = None,
        answer_weight: float = ANSWER_WEIGHT,
        chain_weight: float = CHAIN_WEIGHT,
    ) -> None:
        self.evaluator_id = evaluator_id
        self.rubrics = dict(RUBRICS_BY_TASK if rubrics is None else rubrics)
        total = answer_weight + chain_weight
        if not 0.0 < total <= 1.0 + 1e-9:
            raise ValueError("answer_weight + chain_weight must be in (0, 1]")
        self.answer_weight = answer_weight
        self.chain_weight = chain_weight

    def __call__(self, task: TaskSpec, trajectory: Trajectory) -> EvaluatorVerdict:
        rubric = self.rubrics.get(task.task_id)
        if rubric is None:
            raise KeyError(
                f"no rubric for task_id {task.task_id!r}; this evaluator grades only "
                "the api_orchestration task set"
            )

        answer = _normalize(trajectory.final_answer or "")
        hit_forbidden = [phrase for phrase in rubric.forbidden if _normalize(phrase) in answer]
        missing_facts = [fact.label for fact in rubric.facts if not fact.matched_by(answer)]
        missing_hops = [
            f"{hop.tool}({_render_args(hop.args)})"
            for hop in rubric.hops
            if not hop.matched_by(trajectory)
        ]

        if not trajectory.final_answer or hit_forbidden:
            answer_score = 0.0
        else:
            answer_score = (len(rubric.facts) - len(missing_facts)) / len(rubric.facts)
        chain_score = (len(rubric.hops) - len(missing_hops)) / len(rubric.hops)

        score = self.answer_weight * answer_score + self.chain_weight * chain_score
        passed = answer_score == 1.0 and chain_score == 1.0

        return EvaluatorVerdict(
            evaluator_id=self.evaluator_id,
            passed=passed,
            score=min(1.0, max(0.0, score)),
            rationale=_rationale(
                answer_score, chain_score, missing_facts, missing_hops, hit_forbidden,
                answered=bool(trajectory.final_answer),
            ),
        )


def _render_args(args: dict[str, Any]) -> str:
    return ", ".join(f"{name}={value!r}" for name, value in args.items())


def _rationale(
    answer_score: float,
    chain_score: float,
    missing_facts: list[str],
    missing_hops: list[str],
    hit_forbidden: list[str],
    *,
    answered: bool,
) -> str:
    parts = [f"answer {answer_score:.2f}, chain {chain_score:.2f}"]
    if not answered:
        parts.append("no final answer was produced")
    if hit_forbidden:
        parts.append("answer asserts something contradicted: " + ", ".join(hit_forbidden))
    if missing_facts:
        parts.append("answer is missing: " + ", ".join(missing_facts))
    if missing_hops:
        parts.append("dependent calls never made: " + "; ".join(missing_hops))
    if not missing_facts and not missing_hops and not hit_forbidden and answered:
        parts.append("full chain walked and every required fact reported")
    return "; ".join(parts)
