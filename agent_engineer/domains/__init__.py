"""Domain packages: task sets and evaluators, one package per domain.

Each domain package exports exactly ``get_suite() -> DomainSuite`` and
``get_evaluator() -> TaskEvaluator``. See ``agent_engineer/evaluation/INTERFACE.md``.
Nothing in here is imported by the engine, which is domain-agnostic.
"""
