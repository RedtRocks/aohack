"""Multi-step API orchestration: dependent tool chains over a simulated support API.

Per ``agent_engineer/evaluation/INTERFACE.md`` this package exports exactly two
module-level callables:

* :func:`get_suite` -- a ``DomainSuite`` of :class:`~agent_engineer.evaluation.TaskSpec`.
* :func:`get_evaluator` -- a ``TaskEvaluator``: ``(task, trajectory) -> EvaluatorVerdict``.

Nothing else here is part of the contract. The engine and the harness reach
this domain only through those two functions; this package imports the frozen
schemas and the measurement interface, and nothing from ``agent_engineer.ports``
or the engine's stages.
"""

from agent_engineer.domains.api_orchestration.evaluator import DependencyEvaluator
from agent_engineer.domains.api_orchestration.tasks import build_suite
from agent_engineer.evaluation import DomainSuite, TaskEvaluator

_SUITE = build_suite()
_EVALUATOR = DependencyEvaluator()


def get_suite() -> DomainSuite:
    """The api_orchestration task suite."""
    return _SUITE


def get_evaluator() -> TaskEvaluator:
    """The api_orchestration grader: ``(task, trajectory) -> EvaluatorVerdict``."""
    return _EVALUATOR


__all__ = ["get_suite", "get_evaluator"]
