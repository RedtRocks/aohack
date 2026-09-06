"""Domain packages for the agent-engineer build.

Each subpackage under here is one domain: it exports exactly two module-level
callables, ``get_suite() -> DomainSuite`` and
``get_evaluator() -> TaskEvaluator``, per
``agent_engineer/evaluation/INTERFACE.md``. Nothing in here is imported by the
engine, which is domain-agnostic.
"""
