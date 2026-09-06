"""Evaluation harness, the four metrics, and the results tables.

See ``INTERFACE.md`` in this package for the contract the engine and the three
domain agents build against.
"""

from agent_engineer.evaluation.harness import (
    MIN_RUNS_FOR_RELIABILITY,
    REFUSAL_MARKERS,
    DomainReport,
    DomainSuite,
    Evaluator,
    IterationReport,
    RunOutcome,
    TaskEvaluator,
    TaskResult,
    TaskRunner,
    TaskSpec,
    is_refusal,
)
from agent_engineer.evaluation.metrics import (
    Metric,
    MetricKind,
    mean_of_defined,
    population_variance,
    undefined,
)

__all__ = [
    "MIN_RUNS_FOR_RELIABILITY",
    "REFUSAL_MARKERS",
    "DomainReport",
    "DomainSuite",
    "Evaluator",
    "IterationReport",
    "Metric",
    "MetricKind",
    "RunOutcome",
    "TaskEvaluator",
    "TaskResult",
    "TaskRunner",
    "TaskSpec",
    "is_refusal",
    "mean_of_defined",
    "population_variance",
    "undefined",
]
