"""The five stages of the agent-engineer loop.

Each stage is a protocol plus at least one implementation that needs no model
access, so the whole loop runs deterministically and offline; model-assisted
implementations are drop-in alternatives behind the same protocol.

1. :mod:`~agent_engineer.stages.synthesize` -- goal + tools + evaluator -> AgentSpec
2. :mod:`~agent_engineer.stages.evaluate` -- AgentSpec + task set -> Trajectory per task
3. :mod:`~agent_engineer.stages.diagnose` -- failing trajectories -> Diagnosis
4. :mod:`~agent_engineer.stages.mutate` -- Diagnosis -> one Mutation -> child AgentSpec
5. :mod:`~agent_engineer.stages.select` -- measured delta -> keep or revert
"""
