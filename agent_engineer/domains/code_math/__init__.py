"""The code_math domain: a task suite and an evaluator, and nothing else.

Correctness is decided by execution or exact comparison -- never by a judge --
so the same submission always earns the same score. This package exports
exactly the two callables the harness contract calls for::

    from agent_engineer.domains.code_math import get_suite, get_evaluator

    suite = get_suite()            # -> DomainSuite
    task_evaluator = get_evaluator()  # -> (TaskSpec, Trajectory) -> EvaluatorVerdict

See ``agent_engineer/evaluation/INTERFACE.md`` for the contract this package
conforms to. This domain grades only: it computes no accuracy, cost, or
reliability, and aggregates nothing -- that is the harness's job.
"""

from agent_engineer.domains.code_math.catalogue import DOMAIN, load_suite
from agent_engineer.domains.code_math.evaluator import EVALUATOR_ID, evaluate, evaluate_answer
from agent_engineer.evaluation import DomainSuite, TaskEvaluator

__all__ = ["DOMAIN", "EVALUATOR_ID", "get_suite", "get_evaluator", "evaluate_answer"]


def get_suite() -> DomainSuite:
    """Return the frozen code_math task suite."""
    return load_suite()


def get_evaluator() -> TaskEvaluator:
    """Return the code_math TaskEvaluator: ``(TaskSpec, Trajectory) -> EvaluatorVerdict``."""
    return evaluate
