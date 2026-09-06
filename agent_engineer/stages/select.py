"""Stage 5: keep or revert a mutated spec on the measured delta.

The decision is made on a number that was measured after the mutation, against
the number measured before it, on the same task set. Nothing here inspects the
mutation's plausibility: a well-argued edit that does not move the metric is
reverted, and that is the point of the stage.

The metric itself is not defined here. The loop is handed a scalar per
evaluation run by the measurement side, and this stage only compares two of
them, so improving the metric never means editing the engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class Decision(str, Enum):
    """What the loop did with a mutated spec."""

    ACCEPTED = "accepted"
    REVERTED = "reverted"


@dataclass(frozen=True)
class Verdict:
    """The keep-or-revert outcome, carrying both numbers it was decided on.

    ``before`` is retained alongside ``delta`` deliberately: a delta of +0.10
    means something different from a base of 0.10 than from a base of 0.85, and
    a lineage that reports only deltas cannot be audited.
    """

    decision: Decision
    before: float
    after: float
    reason: str

    @property
    def delta(self) -> float:
        return self.after - self.before

    @property
    def accepted(self) -> bool:
        return self.decision is Decision.ACCEPTED


@runtime_checkable
class SelectionPolicy(Protocol):
    """Decides whether a measured change is worth keeping."""

    def decide(self, before: float, after: float) -> Verdict: ...


class MinimumDeltaPolicy:
    """Keep a mutation only when it improves the metric by at least ``min_delta``.

    ``min_delta`` defaults to a small positive number rather than zero so that
    noise-sized movements do not get locked in as progress. A tie or a
    regression reverts: the parent spec was already measured at that level and
    is the simpler artifact.
    """

    def __init__(self, min_delta: float = 1e-9) -> None:
        if min_delta < 0:
            raise ValueError("min_delta must not be negative; that would keep regressions")
        self._min_delta = min_delta

    def decide(self, before: float, after: float) -> Verdict:
        delta = after - before
        if delta >= self._min_delta:
            return Verdict(
                decision=Decision.ACCEPTED,
                before=before,
                after=after,
                reason=(
                    f"measured {before:.4f} -> {after:.4f} (delta {delta:+.4f}), which clears the "
                    f"{self._min_delta:g} keep threshold"
                ),
            )
        return Verdict(
            decision=Decision.REVERTED,
            before=before,
            after=after,
            reason=(
                f"measured {before:.4f} -> {after:.4f} (delta {delta:+.4f}), which does not clear "
                f"the {self._min_delta:g} keep threshold; reverting to the parent spec"
            ),
        )


class RegressionTolerantPolicy:
    """Keep anything that does not regress, so neutral structural edits can accumulate.

    Useful when the metric is coarse -- a pass rate over a small task set moves
    in visible jumps, and an edit that fixes a real cause can measure flat for a
    generation before the next edit converts it.
    """

    def __init__(self, tolerance: float = 0.0) -> None:
        if tolerance < 0:
            raise ValueError("tolerance must not be negative")
        self._tolerance = tolerance

    def decide(self, before: float, after: float) -> Verdict:
        delta = after - before
        keep = delta >= -self._tolerance
        return Verdict(
            decision=Decision.ACCEPTED if keep else Decision.REVERTED,
            before=before,
            after=after,
            reason=(
                f"measured {before:.4f} -> {after:.4f} (delta {delta:+.4f}); "
                + (
                    f"within the {self._tolerance:g} no-regression tolerance, keeping"
                    if keep
                    else f"a regression beyond the {self._tolerance:g} tolerance, reverting"
                )
            ),
        )
