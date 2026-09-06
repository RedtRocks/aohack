"""The mcp_everything evaluator: grades trajectory tool calls and output against task requirements.

Exported as evaluate matching the harness's TaskEvaluator protocol:
    (TaskSpec, Trajectory) -> EvaluatorVerdict

Grades only: computes no accuracy, cost, or reliability, and aggregates nothing.
Counting is the harness's job.
"""

from __future__ import annotations

import re
from typing import Any

from agent_engineer.evaluation import TaskSpec
from agent_engineer.schemas import EvaluatorVerdict, Trajectory

__all__ = ["EVALUATOR_ID", "evaluate"]

EVALUATOR_ID = "mcp_everything-v1"

_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Normalize text for robust comparison."""
    return _WHITESPACE.sub(" ", text.lower()).strip()


def evaluate(task: TaskSpec, trajectory: Trajectory) -> EvaluatorVerdict:
    """Grade one trajectory against an mcp_everything task.

    Never raises. Grades only: never computes accuracy, cost, or aggregates.
    Checks:
    1. Tool invocation: Did the agent call the required MCP tool without error?
    2. Answer correctness: Does the final answer include the expected output?
    """
    if trajectory.final_answer is None:
        return EvaluatorVerdict(
            evaluator_id=EVALUATOR_ID,
            passed=False,
            score=0.0,
            rationale="no final answer was produced",
        )

    required_tools = tuple(task.metadata.get("required_tools", ()))
    expected_fragment = task.metadata.get("expected_fragment", "")
    expected = task.expected or ""

    # Check tool execution
    matched_call = None
    for call in trajectory.tool_calls:
        if call.tool_name in required_tools:
            matched_call = call
            break

    tool_ok = False
    tool_err = None
    if matched_call is not None:
        if matched_call.error is None:
            tool_ok = True
        else:
            tool_err = matched_call.error

    # Check answer
    norm_answer = _normalize(trajectory.final_answer)
    norm_fragment = _normalize(expected_fragment)
    norm_expected = _normalize(expected)
    answer_ok = (norm_fragment in norm_answer) or (norm_expected in norm_answer)

    # When tool is called successfully and answer is correct
    if tool_ok and answer_ok:
        return EvaluatorVerdict(
            evaluator_id=EVALUATOR_ID,
            passed=True,
            score=1.0,
            rationale=f"Tool {matched_call.tool_name!r} executed successfully and final answer matches expected output.",
        )

    # Partial credit: tool succeeded but answer was incomplete/missing facts
    if tool_ok and not answer_ok:
        return EvaluatorVerdict(
            evaluator_id=EVALUATOR_ID,
            passed=False,
            score=0.3,
            rationale=f"Tool {matched_call.tool_name!r} succeeded, but answer did not incorporate the result (expected fragment {expected_fragment!r}).",
        )

    # Partial credit: answer matched reference facts (e.g. from prior knowledge or generic solver)
    # but the required tool was not invoked or failed. This provides an evaluation gradient.
    if not tool_ok and answer_ok:
        if matched_call is not None:
            rationale = f"Tool {matched_call.tool_name!r} failed ({tool_err}), though answer matched expected facts."
        else:
            rationale = f"Required tool from {required_tools!r} was not invoked, though answer matched expected facts."
        return EvaluatorVerdict(
            evaluator_id=EVALUATOR_ID,
            passed=False,
            score=0.7,
            rationale=rationale,
        )

    # Neither passed
    if matched_call is not None:
        rationale = f"Tool {matched_call.tool_name!r} failed ({tool_err}) and answer was incorrect."
    else:
        rationale = f"Required tool from {required_tools!r} was not invoked and answer was incorrect."
    return EvaluatorVerdict(
        evaluator_id=EVALUATOR_ID,
        passed=False,
        score=0.0,
        rationale=rationale,
    )
