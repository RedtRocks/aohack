"""The structured-extraction domain: messy documents with frozen ground truth.

Exports exactly the two callables the harness needs, per
``agent_engineer/evaluation/INTERFACE.md``:

* :func:`get_suite` -- a :class:`~agent_engineer.evaluation.DomainSuite`.
* :func:`get_evaluator` -- a :class:`~agent_engineer.evaluation.TaskEvaluator`.

Nothing else in this package is part of the engine-facing surface. The engine
must never import :mod:`agent_engineer.domains.extraction.tasks` or
:mod:`agent_engineer.domains.extraction.evaluator` directly.
"""

from __future__ import annotations

from agent_engineer.domains.extraction.evaluator import EVALUATOR, ExtractionEvaluator
from agent_engineer.domains.extraction.tasks import DOMAIN, build_suite
from agent_engineer.evaluation import DomainSuite, TaskEvaluator

__all__ = ["DOMAIN", "get_suite", "get_evaluator"]


def get_suite() -> DomainSuite:
    """The structured-extraction task set, rendered as a :class:`DomainSuite`."""
    return build_suite()


def get_evaluator() -> TaskEvaluator:
    """The structured-extraction evaluator, as a ``(TaskSpec, Trajectory) -> EvaluatorVerdict``."""
    return EVALUATOR
