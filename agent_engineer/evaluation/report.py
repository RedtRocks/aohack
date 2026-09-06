"""Results tables.

Two rules are enforced *here*, in the rendering layer, rather than left to the
discipline of whoever writes the demo script:

* **No improvement may be rendered without its before number.** An
  :class:`Improvement` cannot be constructed from an after-value alone, so
  "accuracy 0.82!" with no baseline is not expressible.
* **No accuracy figure may be rendered without its paired cost.** More retries
  always buys accuracy, so an accuracy number on its own is not a result. The
  accuracy and cost cells are produced by a single call that takes both.

Both are enforced by the shapes of the functions, not by a comment asking the
caller to remember. An undefined :class:`Metric` renders as ``n/a`` with its
reason available for a footnote -- never as ``0``.
"""

from __future__ import annotations

from collections.abc import Sequence

from agent_engineer.evaluation.harness import DomainReport, IterationReport
from agent_engineer.evaluation.metrics import Metric, MetricKind

__all__ = [
    "UNDEFINED_CELL",
    "Improvement",
    "format_metric",
    "format_accuracy_with_cost",
    "render_iteration_table",
    "render_progress_table",
    "render_report",
]

UNDEFINED_CELL = "n/a"
"""What an undefined metric renders as. Deliberately not ``0`` and not blank."""

_PRECISION: dict[MetricKind, int] = {
    "accuracy": 3,
    "reliability": 4,
    "cost": 1,
    "speed": 2,
}

_UNITS: dict[MetricKind, str] = {
    "accuracy": "",
    "reliability": "",
    "cost": " tok",
    "speed": " s",
}


def format_metric(metric: Metric) -> str:
    """Render one metric cell. Undefined metrics render as ``n/a``, never as a number."""
    if not metric.is_defined:
        return UNDEFINED_CELL
    digits = _PRECISION.get(metric.kind, 3)
    return f"{metric.require():.{digits}f}{_UNITS.get(metric.kind, '')}"


class Improvement:
    """A change in a metric, which cannot exist without the number it improved on.

    There is no way to build one of these from an after-value alone: ``before``
    is a required positional argument. That is the point -- an improvement
    rendered without its baseline is a claim, not a measurement.
    """

    def __init__(self, kind: MetricKind, before: Metric, after: Metric) -> None:
        if before.kind != kind or after.kind != kind:
            raise ValueError(f"both metrics must be of kind {kind!r}")
        self.kind = kind
        self.before = before
        self.after = after

    @property
    def is_renderable(self) -> bool:
        """False when either end is undefined: a delta needs two real numbers."""
        return self.before.is_defined and self.after.is_defined

    @property
    def delta(self) -> float | None:
        if not self.is_renderable:
            return None
        return self.after.require() - self.before.require()

    def render(self) -> str:
        """``before -> after (+delta)``. Never the after-value on its own."""
        if not self.is_renderable:
            missing = "before" if not self.before.is_defined else "after"
            return f"{UNDEFINED_CELL} (no {missing} number)"
        delta = self.delta
        assert delta is not None
        digits = _PRECISION.get(self.kind, 3)
        return (
            f"{format_metric(self.before)} -> {format_metric(self.after)} "
            f"({delta:+.{digits}f})"
        )


def format_accuracy_with_cost(accuracy: Metric, cost: Metric) -> tuple[str, str]:
    """Render accuracy and its cost together. Neither is available without the other.

    Both cells come out of one call, so there is no code path that produces an
    accuracy string while the cost is still sitting unrendered somewhere.
    """
    if accuracy.kind != "accuracy":
        raise ValueError(f"expected an accuracy metric, got {accuracy.kind!r}")
    if cost.kind != "cost":
        raise ValueError(
            "accuracy may not be rendered without its paired cost; "
            f"got a {cost.kind!r} metric instead"
        )
    return format_metric(accuracy), format_metric(cost)


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)).rstrip(),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)).rstrip()
        for row in rows
    )
    return "\n".join(lines)


def _domain_row(report: DomainReport) -> list[str]:
    accuracy_cell, cost_cell = format_accuracy_with_cost(report.accuracy, report.cost)
    return [
        report.domain,
        accuracy_cell,
        cost_cell,
        format_metric(report.reliability),
        format_metric(report.speed),
        str(report.refusal_count),
        str(report.task_count * report.repeats),
    ]


def render_iteration_table(report: IterationReport) -> str:
    """One iteration: every domain's four metrics, plus the cross-domain averages.

    Accuracy and cost sit side by side in every row. The average row is computed
    over defined metrics only, so a domain that never ran leaves a gap rather
    than pulling the mean towards zero.
    """
    headers = ["domain", "accuracy", "cost", "reliability", "speed", "refusals", "runs"]
    rows = [_domain_row(domain) for domain in report.domains]

    mean_accuracy, mean_cost = format_accuracy_with_cost(
        report.mean_accuracy, report.mean_cost
    )
    defined = report.mean_accuracy.sample_size
    rows.append(
        [
            f"MEAN ({defined}/{len(report.domains)} defined)",
            mean_accuracy,
            mean_cost,
            format_metric(report.mean_reliability),
            format_metric(report.mean_speed),
            str(sum(domain.refusal_count for domain in report.domains)),
            "",
        ]
    )

    title = f"iteration {report.iteration}  (spec {report.spec_id})"
    body = _table(headers, rows)
    footnotes = _undefined_footnotes(report)
    return f"{title}\n{body}" + (f"\n\n{footnotes}" if footnotes else "")


def _undefined_footnotes(report: IterationReport) -> str:
    notes = [
        f"  {domain.domain}.{kind}: {metric.reason}"
        for domain in report.domains
        for kind, metric in domain.metrics.items()
        if not metric.is_defined
    ]
    if not notes:
        return ""
    return "undefined (excluded from every average, never counted as zero):\n" + "\n".join(notes)


def render_progress_table(reports: Sequence[IterationReport]) -> str:
    """Per domain, the first iteration against the last -- before and after, together.

    Requires at least two iterations. With one there is no before number, and an
    after-value on its own is not an improvement.
    """
    if len(reports) < 2:
        raise ValueError(
            "a progress table needs at least two iterations; with one there is no "
            "before number, and no improvement may be rendered without it"
        )
    ordered = sorted(reports, key=lambda report: report.iteration)
    first, last = ordered[0], ordered[-1]

    headers = ["domain", "accuracy (before -> after)", "cost (before -> after)", "reliability"]
    rows: list[list[str]] = []
    for domain in last.domains:
        try:
            baseline = first.domain(domain.domain)
        except KeyError:
            continue
        # Accuracy and cost are always emitted as a pair, so an accuracy gain
        # can never be shown without what it cost to buy.
        rows.append(
            [
                domain.domain,
                Improvement("accuracy", baseline.accuracy, domain.accuracy).render(),
                Improvement("cost", baseline.cost, domain.cost).render(),
                Improvement("reliability", baseline.reliability, domain.reliability).render(),
            ]
        )
    rows.append(
        [
            "MEAN",
            Improvement("accuracy", first.mean_accuracy, last.mean_accuracy).render(),
            Improvement("cost", first.mean_cost, last.mean_cost).render(),
            Improvement("reliability", first.mean_reliability, last.mean_reliability).render(),
        ]
    )
    title = f"progress: iteration {first.iteration} -> {last.iteration}"
    return f"{title}\n{_table(headers, rows)}"


def render_report(reports: Sequence[IterationReport]) -> str:
    """The full results write-up: every iteration, then the before/after summary."""
    if not reports:
        raise ValueError("nothing to report: no iterations were run")
    ordered = sorted(reports, key=lambda report: report.iteration)
    sections = [render_iteration_table(report) for report in ordered]
    if len(ordered) >= 2:
        sections.append(render_progress_table(ordered))
    return "\n\n".join(sections) + "\n"
