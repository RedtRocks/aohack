"""A deterministic, offline stand-in for a support-operations HTTP API.

Every tool here is a pure function of the frozen dataset below: no network, no
clock, no keys, no randomness. Two calls with the same arguments return equal
results forever, which is what makes the task set gradeable.

The point of this module is *dependency*. The identifiers each tool needs are
produced by other tools and by nothing else: a ``customer_id`` only ever comes
out of :meth:`SimulatedSupportAPI.find_customer`, an ``order_id`` only out of
:meth:`SimulatedSupportAPI.list_orders`, and a ``shipment_id``, ``sku`` or
``warehouse_id`` only out of :meth:`SimulatedSupportAPI.get_order`. None of it
is guessable from a task prompt, so an agent that cannot carry a value from one
call into the next cannot answer at all.

Three shapes of hop appear deliberately:

* **Straight chains** -- email to customer to order to shipment.
* **Narrowing** -- ``get_order`` returns ``shipment_id: null`` for a backordered
  order, and that null is the signal to ask inventory rather than the carrier.
* **Recoverable faults** -- ``track_shipment`` fails for one carrier and
  ``compute_refund`` refuses an undelivered order. Both failures name the tool
  that does work, so the recovery is discoverable rather than guessable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


class ToolError(RuntimeError):
    """A simulated API failure.

    Raised for bad arguments, missing records, and the injected faults. A
    caller records this as ``ToolCall.error`` and keeps going, because several
    tasks are only solvable by recovering from one.
    """


@dataclass(frozen=True)
class Customer:
    customer_id: str
    email: str
    name: str
    tier: str
    region: str


@dataclass(frozen=True)
class Order:
    order_id: str
    customer_id: str
    status: str
    total_cents: int
    warehouse_id: str
    line_items: tuple[tuple[str, int], ...]
    shipment_id: str | None
    delivered_days_ago: int | None


@dataclass(frozen=True)
class Shipment:
    shipment_id: str
    carrier: str
    state: str
    last_scan_city: str
    eta_days: int
    events: tuple[tuple[str, str, str], ...]
    """(when, state, city), oldest first."""


CUSTOMERS: tuple[Customer, ...] = (
    Customer("c_ada", "ada@lovelace.io", "Ada Lovelace", "gold", "EU"),
    Customer("c_grace", "grace@hopper.mil", "Grace Hopper", "platinum", "US"),
    Customer("c_alan", "alan@turing.uk", "Alan Turing", "standard", "EU"),
    Customer("c_katherine", "katherine@johnson.nasa", "Katherine Johnson", "gold", "US"),
    Customer("c_linus", "linus@torvalds.fi", "Linus Torvalds", "standard", "EU"),
)

ORDERS: tuple[Order, ...] = (
    Order("o_1001", "c_ada", "shipped", 24500, "w_eu1", (("sku_kbd", 1),), "s_501", None),
    Order("o_1002", "c_ada", "delivered", 8900, "w_eu1", (("sku_mouse", 2),), "s_502", 12),
    Order("o_1007", "c_ada", "cancelled", 5000, "w_eu1", (("sku_kbd", 1),), None, None),
    Order("o_1003", "c_grace", "backordered", 129900, "w_us2", (("sku_gpu", 1),), None, None),
    Order("o_1004", "c_grace", "shipped", 4500, "w_us1", (("sku_cable", 3),), "s_503", None),
    Order("o_1005", "c_alan", "delivered", 15000, "w_eu2", (("sku_desk", 1),), "s_504", 45),
    Order("o_1006", "c_katherine", "backordered", 259800, "w_us1", (("sku_gpu", 2),), None, None),
)

SHIPMENTS: tuple[Shipment, ...] = (
    Shipment(
        "s_501", "NORDPOST", "in_transit", "Rotterdam", 2,
        (("d-3", "accepted", "Dublin"), ("d-1", "in_transit", "Rotterdam")),
    ),
    Shipment(
        "s_502", "DHL_SIM", "delivered", "Berlin", 0,
        (("d-14", "accepted", "Berlin"), ("d-12", "delivered", "Berlin")),
    ),
    Shipment(
        # track_shipment is deliberately broken for this carrier; the stored
        # event log carries the same facts and is the intended recovery.
        "s_503", "FLEETWAY", "in_transit", "Denver", 4,
        (("d-2", "accepted", "Phoenix"), ("d-1", "in_transit", "Denver")),
    ),
    Shipment(
        "s_504", "DHL_SIM", "delivered", "Manchester", 0,
        (("d-47", "accepted", "Leeds"), ("d-45", "delivered", "Manchester")),
    ),
)

BROKEN_CARRIERS = frozenset({"FLEETWAY"})

WAREHOUSES: dict[str, tuple[str, ...]] = {
    "EU": ("w_eu1", "w_eu2"),
    "US": ("w_us1", "w_us2", "w_us3"),
}

INVENTORY: dict[tuple[str, str], tuple[int, int]] = {
    # (sku, warehouse_id) -> (on_hand, restock_days)
    ("sku_gpu", "w_us1"): (0, 5),
    ("sku_gpu", "w_us2"): (0, 14),
    ("sku_gpu", "w_us3"): (3, 0),
    ("sku_kbd", "w_eu1"): (42, 0),
    ("sku_mouse", "w_eu1"): (7, 0),
    ("sku_cable", "w_us1"): (100, 0),
    ("sku_desk", "w_eu2"): (2, 0),
}

REFUND_POLICY: dict[tuple[str, str], tuple[int, int]] = {
    # (tier, region) -> (window_days, auto_approve_cents)
    ("gold", "EU"): (30, 10000),
    ("gold", "US"): (30, 12000),
    ("platinum", "EU"): (60, 50000),
    ("platinum", "US"): (60, 50000),
    ("standard", "EU"): (14, 5000),
    ("standard", "US"): (14, 5000),
}


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"{field} must be a non-empty string, got {value!r}")
    return value.strip()


class SimulatedSupportAPI:
    """The tool surface. Stateless: instances exist only to be cheap to pass around."""

    # -- lookup -------------------------------------------------------------

    def find_customer(self, email: str) -> dict[str, Any]:
        """Resolve an email to a customer record. The only source of customer_id."""
        wanted = _require_str(email, "email").lower()
        for customer in CUSTOMERS:
            if customer.email == wanted:
                return {
                    "customer_id": customer.customer_id,
                    "name": customer.name,
                    "tier": customer.tier,
                    "region": customer.region,
                }
        raise ToolError(f"no customer found for email {wanted!r}")

    def list_orders(self, customer_id: str, status: str | None = None) -> list[dict[str, Any]]:
        """Orders for a customer, optionally filtered by status.

        Returns an empty list rather than an error when nothing matches: an
        empty result is an answer, and dropping the filter is how you widen.
        """
        wanted = _require_str(customer_id, "customer_id")
        if not any(customer.customer_id == wanted for customer in CUSTOMERS):
            raise ToolError(f"unknown customer_id {wanted!r}; resolve it with find_customer first")
        if status is not None:
            status = _require_str(status, "status").lower()
        return [
            {"order_id": order.order_id, "status": order.status, "total_cents": order.total_cents}
            for order in ORDERS
            if order.customer_id == wanted and (status is None or order.status == status)
        ]

    def get_order(self, order_id: str) -> dict[str, Any]:
        """Full order detail. The only source of sku, warehouse_id, and shipment_id.

        ``shipment_id`` is ``None`` for anything not yet handed to a carrier;
        that null is the fork in the road between tracking and inventory.
        """
        wanted = _require_str(order_id, "order_id")
        for order in ORDERS:
            if order.order_id == wanted:
                return {
                    "order_id": order.order_id,
                    "customer_id": order.customer_id,
                    "status": order.status,
                    "total_cents": order.total_cents,
                    "warehouse_id": order.warehouse_id,
                    "line_items": [{"sku": sku, "qty": qty} for sku, qty in order.line_items],
                    "shipment_id": order.shipment_id,
                }
        raise ToolError(f"unknown order_id {wanted!r}; list_orders returns valid ids")

    # -- logistics ----------------------------------------------------------

    def track_shipment(self, shipment_id: str) -> dict[str, Any]:
        """Live carrier tracking. Fails for carriers whose feed is down."""
        shipment = self._shipment(shipment_id)
        if shipment.carrier in BROKEN_CARRIERS:
            raise ToolError(
                f"carrier {shipment.carrier} tracking feed is unavailable for "
                f"{shipment.shipment_id}; call list_shipment_events for the stored scan log"
            )
        return {
            "shipment_id": shipment.shipment_id,
            "carrier": shipment.carrier,
            "state": shipment.state,
            "last_scan_city": shipment.last_scan_city,
            "eta_days": shipment.eta_days,
        }

    def list_shipment_events(self, shipment_id: str) -> list[dict[str, str]]:
        """The stored scan log. Always available, oldest event first."""
        shipment = self._shipment(shipment_id)
        return [
            {"when": when, "state": state, "city": city} for when, state, city in shipment.events
        ]

    def _shipment(self, shipment_id: str) -> Shipment:
        wanted = _require_str(shipment_id, "shipment_id")
        for shipment in SHIPMENTS:
            if shipment.shipment_id == wanted:
                return shipment
        raise ToolError(
            f"unknown shipment_id {wanted!r}; get_order returns the shipment_id for an order, "
            "and it is null when the order has not shipped"
        )

    # -- supply -------------------------------------------------------------

    def list_warehouses(self, region: str) -> list[str]:
        """Warehouse ids serving a region. The customer's region comes from find_customer."""
        wanted = _require_str(region, "region").upper()
        if wanted not in WAREHOUSES:
            raise ToolError(f"unknown region {wanted!r}; known regions are EU, US")
        return list(WAREHOUSES[wanted])

    def get_inventory(self, sku: str, warehouse_id: str) -> dict[str, Any]:
        """Stock for one sku at one warehouse."""
        sku = _require_str(sku, "sku")
        warehouse_id = _require_str(warehouse_id, "warehouse_id")
        record = INVENTORY.get((sku, warehouse_id))
        if record is None:
            raise ToolError(f"no inventory record for sku {sku!r} at warehouse {warehouse_id!r}")
        on_hand, restock_days = record
        return {
            "sku": sku,
            "warehouse_id": warehouse_id,
            "on_hand": on_hand,
            "restock_days": restock_days,
        }

    # -- money --------------------------------------------------------------

    def get_refund_policy(self, tier: str, region: str) -> dict[str, Any]:
        """Refund terms for a tier in a region. Both come from find_customer."""
        tier = _require_str(tier, "tier").lower()
        region = _require_str(region, "region").upper()
        record = REFUND_POLICY.get((tier, region))
        if record is None:
            raise ToolError(f"no refund policy for tier {tier!r} in region {region!r}")
        window_days, auto_approve_cents = record
        return {
            "tier": tier,
            "region": region,
            "window_days": window_days,
            "auto_approve_cents": auto_approve_cents,
        }

    def compute_refund(self, order_id: str, policy_window_days: int) -> dict[str, Any]:
        """Refund for a delivered order under a policy window.

        Refuses anything not delivered, which is a fact only ``get_order`` knows.
        """
        wanted = _require_str(order_id, "order_id")
        if isinstance(policy_window_days, bool) or not isinstance(policy_window_days, int):
            raise ToolError(
                f"policy_window_days must be an int, got {policy_window_days!r}; "
                "get_refund_policy returns it as window_days"
            )
        for order in ORDERS:
            if order.order_id != wanted:
                continue
            if order.status != "delivered" or order.delivered_days_ago is None:
                raise ToolError(
                    f"order {wanted} is {order.status}, not delivered; refunds apply to "
                    "delivered orders only"
                )
            within = order.delivered_days_ago <= policy_window_days
            return {
                "order_id": order.order_id,
                "days_since_delivery": order.delivered_days_ago,
                "policy_window_days": policy_window_days,
                "within_window": within,
                "refund_cents": order.total_cents if within else 0,
            }
        raise ToolError(f"unknown order_id {wanted!r}; list_orders returns valid ids")


TOOL_NAMES: tuple[str, ...] = (
    "find_customer",
    "list_orders",
    "get_order",
    "track_shipment",
    "list_shipment_events",
    "list_warehouses",
    "get_inventory",
    "get_refund_policy",
    "compute_refund",
)


def tool_registry(api: SimulatedSupportAPI | None = None) -> dict[str, Callable[..., Any]]:
    """Name-to-callable map for the whole tool surface."""
    api = api or SimulatedSupportAPI()
    return {name: getattr(api, name) for name in TOOL_NAMES}
