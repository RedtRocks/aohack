"""Domain packages for the agent-engineer build.

Each subpackage under here is one domain: it exports exactly two module-level
callables, ``get_suite() -> DomainSuite`` and
``get_evaluator() -> TaskEvaluator``, per
``agent_engineer/evaluation/INTERFACE.md``. Nothing in here is imported by the
engine, which is domain-agnostic.

This module is the registry that lets a caller reach any domain by name,
uniformly, without having to know each subpackage's import path:

    from agent_engineer.domains import get_suite, get_evaluator

    suite = get_suite("code_math")
    evaluator = get_evaluator("code_math")

It exists for callers outside the engine (the harness driver, the demo, tests
proving a non-zero baseline); the engine itself never imports this module or
any name below.
"""

from __future__ import annotations

from agent_engineer.domains import api_orchestration, code_math, extraction, mcp_everything
from agent_engineer.evaluation import DomainSuite, TaskEvaluator

__all__ = ["DOMAIN_NAMES", "get_suite", "get_evaluator"]

_DOMAINS = {
    "code_math": code_math,
    "api_orchestration": api_orchestration,
    "extraction": extraction,
    "mcp_everything": mcp_everything,
}

DOMAIN_NAMES: tuple[str, ...] = tuple(_DOMAINS)
"""The registered domain names, in registration order."""


def get_suite(domain: str) -> DomainSuite:
    """Return the named domain's task suite."""
    return _DOMAINS[domain].get_suite()


def get_evaluator(domain: str) -> TaskEvaluator:
    """Return the named domain's evaluator: ``(TaskSpec, Trajectory) -> EvaluatorVerdict``."""
    return _DOMAINS[domain].get_evaluator()
