"""Metric primitives for the evaluator.

The one idea in this module is that **a metric with an empty denominator is
undefined, not zero**. :class:`Metric` can therefore hold no value at all, and
:func:`mean_of_defined` skips those rather than folding a zero into the average.
That is the whole reason metrics are wrapped in an object instead of passed
around as bare floats: a bare float has nowhere to put "we never measured this",
so it gets written down as ``0.0`` and quietly drags every aggregate that
touches it downwards.

Nothing here knows what a Trajectory is. Aggregation lives in
:mod:`agent_engineer.evaluation.harness`; presentation lives in
:mod:`agent_engineer.evaluation.report`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, model_validator

__all__ = [
    "MetricKind",
    "Metric",
    "undefined",
    "mean_of_defined",
    "population_variance",
]

MetricKind = Literal["accuracy", "reliability", "cost", "speed"]
"""The four metrics reported per domain per iteration."""

NonEmptyStr = Annotated[str, Field(min_length=1)]


class Metric(BaseModel):
    """One measured number, or an explicit absence of one.

    ``value is None`` means undefined: the denominator was empty, so there is no
    number to report and none may be invented. An undefined metric carries
    ``reason`` explaining what was missing, which is what the report renders in
    the cell instead of a figure.

    Construct defined metrics with the constructor and undefined ones with
    :func:`undefined`; the validators make the two states impossible to confuse.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: MetricKind
    value: float | None = Field(
        default=None, description="The measured value, or None when undefined."
    )
    sample_size: NonNegativeInt = Field(
        default=0, description="Size of the denominator this value was computed over."
    )
    reason: NonEmptyStr | None = Field(
        default=None, description="Why the metric is undefined. Set if and only if value is None."
    )

    @model_validator(mode="after")
    def _value_xor_reason(self) -> Self:
        if self.value is None and self.reason is None:
            raise ValueError("an undefined Metric must carry a reason")
        if self.value is not None and self.reason is not None:
            raise ValueError("a defined Metric must not carry an undefined-reason")
        if self.value is not None and self.sample_size == 0:
            raise ValueError(
                f"{self.kind} has a value but an empty denominator; "
                "an empty denominator is undefined, not zero"
            )
        return self

    @property
    def is_defined(self) -> bool:
        return self.value is not None

    def require(self) -> float:
        """Return the value, or raise if undefined. For callers that cannot proceed without one."""
        if self.value is None:
            raise ValueError(f"{self.kind} is undefined: {self.reason}")
        return self.value


def undefined(kind: MetricKind, reason: str) -> Metric:
    """An explicitly undefined metric. Excluded from every average."""
    return Metric(kind=kind, value=None, sample_size=0, reason=reason)


def mean_of_defined(metrics: Iterable[Metric], *, kind: MetricKind) -> Metric:
    """Average the defined metrics, excluding undefined ones from the denominator.

    Undefined inputs contribute neither a value nor a slot: three domains where
    one never ran average over two, not over three-with-a-zero. If every input
    is undefined the result is undefined too -- there is still nothing to report.
    """
    values = [metric.require() for metric in metrics if metric.is_defined]
    if not values:
        return undefined(kind, f"no domain reported a defined {kind}")
    return Metric(kind=kind, value=sum(values) / len(values), sample_size=len(values))


def population_variance(values: Sequence[float]) -> float:
    """Population variance of ``values``. Requires at least two points."""
    if len(values) < 2:
        raise ValueError("variance needs at least two observations")
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)
