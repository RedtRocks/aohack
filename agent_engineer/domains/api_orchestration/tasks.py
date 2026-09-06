"""The task suite for the multi-step API orchestration domain.

Every task here is unsolvable in one shot. The prompt names a human-readable
handle -- an email address -- and nothing else the API accepts, so the first
call is forced and every call after it consumes an identifier that only the
previous call could have produced. Threading state is the whole exercise.

The set ranges on purpose. A run that passes everything tells the engineer loop
nothing to improve; a run that fails everything gives it no gradient. So there
are two-hop warmups a weak agent can land, four- to six-hop chains it will not,
and in between the cases that separate carrying state from guessing:

* ``t_backorder_restock`` and ``t_soonest_warehouse`` **narrow** -- a null
  ``shipment_id`` partway through the chain redirects the rest of it.
* ``t_carrier_outage`` and ``t_refund_not_delivered`` **fault** -- a tool call
  fails and the error text names the way around it.
* ``t_no_orders`` and ``t_widen_after_empty`` **come back empty** -- the honest
  answer is "none", and in one case the recovery is to drop the filter and ask
  again.

Per ``agent_engineer/evaluation/INTERFACE.md``, each :class:`TaskSpec` carries
its fixtures in ``metadata`` -- a JSON-schema-shaped tool catalogue plus the
one callable that dispatches a call against the simulated API -- because
``metadata`` is opaque to the evaluator and to the engine's own accounting.
Nothing here is imported by the engine directly; a runner reaches this domain
only through ``TaskSpec.metadata``.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from agent_engineer.domains.api_orchestration.api import TOOL_NAMES, SimulatedSupportAPI, ToolError
from agent_engineer.evaluation import DomainSuite, TaskSpec

DOMAIN = "api_orchestration"

_TYPE_NAMES = {str: "string", int: "integer", "str": "string", "int": "integer"}


def _json_type(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return "string"
    text = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", str(annotation))
    text = text.split("|")[0].strip()
    return _TYPE_NAMES.get(text, "string")


def _tool_schema(name: str, func: Callable[..., Any]) -> dict[str, Any]:
    """A JSON-schema-shaped catalogue entry, matching the engine's ToolSchema.parameters."""
    signature = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for parameter_name, parameter in signature.parameters.items():
        properties[parameter_name] = {"type": _json_type(parameter.annotation)}
        if parameter.default is inspect.Parameter.empty:
            required.append(parameter_name)
    return {
        "name": name,
        "description": inspect.getdoc(func) or "",
        "parameters": {"type": "object", "properties": properties, "required": required},
    }


def _make_call_tool(api: SimulatedSupportAPI, allowed: tuple[str, ...]) -> Callable[[str, dict[str, Any]], Any]:
    def call_tool(tool_name: str, args: dict[str, Any]) -> Any:
        if tool_name not in allowed:
            raise ToolError(
                f"tool {tool_name!r} is not available for this task; available: {', '.join(allowed)}"
            )
        if not isinstance(args, dict):
            raise ToolError(f"args must be a dict, got {type(args).__name__}")
        func = getattr(api, tool_name)
        try:
            return func(**args)
        except ToolError:
            raise
        except TypeError as exc:
            raise ToolError(f"bad arguments for {tool_name}: {exc}") from exc

    return call_tool


def _build_task(
    task_id: str,
    prompt: str,
    *,
    difficulty: str,
    min_hops: int,
    api: SimulatedSupportAPI,
    tools: tuple[str, ...] = TOOL_NAMES,
) -> TaskSpec:
    schemas = tuple(_tool_schema(name, getattr(api, name)) for name in tools)
    return TaskSpec(
        task_id=task_id,
        domain=DOMAIN,
        prompt=prompt,
        metadata={
            "difficulty": difficulty,
            "min_hops": min_hops,
            "tool_schemas": schemas,
            "call_tool": _make_call_tool(api, tools),
        },
    )


def build_suite(api: SimulatedSupportAPI | None = None) -> DomainSuite:
    """Construct the suite fresh. ``get_suite`` hands back one canonical instance of this."""
    api = api or SimulatedSupportAPI()
    tasks = (
        _build_task(
            "t_refund_window",
            "The customer with email ada@lovelace.io is asking how long she has to return "
            "something. Report her loyalty tier and the refund window in days that applies "
            "to her.",
            difficulty="easy",
            min_hops=2,
            api=api,
        ),
        _build_task(
            "t_last_scan",
            "Where did alan@turing.uk's order end up? Report the order id and the city of its "
            "most recent carrier scan.",
            difficulty="medium",
            min_hops=4,
            api=api,
        ),
        _build_task(
            "t_track_simple",
            "ada@lovelace.io wants an update on her shipped order. Report the order id and "
            "the city of its last carrier scan.",
            difficulty="medium",
            min_hops=4,
            api=api,
        ),
        _build_task(
            "t_backorder_restock",
            "grace@hopper.mil has an order that has not shipped. Find out how many days until "
            "the item on it is back in stock at the warehouse the order is assigned to, and "
            "report that number of days.",
            difficulty="hard",
            min_hops=4,
            api=api,
        ),
        _build_task(
            "t_carrier_outage",
            "Support needs the current state and last known city of the parcel for "
            "grace@hopper.mil's shipped order. Live tracking may be down for some carriers; "
            "if it is, get the facts another way rather than reporting failure.",
            difficulty="hard",
            min_hops=4,
            api=api,
        ),
        _build_task(
            "t_no_orders",
            "How many orders does linus@torvalds.fi have on file? If there are none, say so "
            "plainly.",
            difficulty="easy",
            min_hops=2,
            api=api,
        ),
        _build_task(
            "t_widen_after_empty",
            "Does alan@turing.uk have any shipped orders right now? If he does not, report "
            "the order id and status of the order he does have.",
            difficulty="medium",
            min_hops=3,
            api=api,
        ),
        _build_task(
            "t_refund_amount",
            "ada@lovelace.io wants to return her delivered order. Using the refund policy that "
            "applies to her, report the refund amount in cents and whether it clears the "
            "automatic-approval threshold.",
            difficulty="expert",
            min_hops=5,
            api=api,
        ),
        _build_task(
            "t_refund_not_delivered",
            "Can grace@hopper.mil be refunded for her shipped order today? Answer yes or no "
            "and give the reason.",
            difficulty="hard",
            min_hops=4,
            api=api,
        ),
        _build_task(
            "t_soonest_warehouse",
            "katherine@johnson.nasa's order is on backorder. Across every warehouse in her "
            "region, find which one can fulfil the item soonest and report that warehouse id "
            "and how soon.",
            difficulty="expert",
            min_hops=6,
            api=api,
        ),
        _build_task(
            "t_unknown_customer",
            "Look up the account for nobody@nowhere.example and report her loyalty tier. If "
            "the account does not exist, say so instead of guessing.",
            difficulty="medium",
            min_hops=1,
            api=api,
        ),
        _build_task(
            "t_open_order_value",
            "Total the value in cents of every order for ada@lovelace.io that was not "
            "cancelled, and report that total.",
            difficulty="medium",
            min_hops=2,
            api=api,
        ),
    )
    return DomainSuite(domain=DOMAIN, tasks=tasks)
