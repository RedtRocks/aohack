"""Stage 1: synthesize an :class:`AgentSpec` from a goal, tool schemas, and an evaluator.

The synthesized system prompt is assembled from material the *caller* supplies
-- the goal string, the tool schemas, and an optional prose description of what
the evaluator rewards. The engine contributes only structural scaffolding that
is true of any agent in any domain ("call a tool or give a final answer", "do
not invent a tool that is not listed"). There is no domain vocabulary here, and
adding any would invalidate the result this project is claiming.

A :class:`~agent_engineer.ports.TaskEvaluator` is a plain grading callable, so
this stage does not depend on it structurally: callers identify it by
``evaluator_id`` and may pass ``criteria`` prose alongside, but nothing here
inspects or calls the evaluator itself.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent_engineer.ports import TextGenerator, ToolSchema
from agent_engineer.schemas import (
    AgentSpec,
    MemoryConfig,
    MemoryKind,
    OrchestrationStrategy,
    StoppingConditions,
)

DEFAULT_MAX_STEPS = 8


@runtime_checkable
class SpecSynthesizer(Protocol):
    """Turns a goal into a starting :class:`AgentSpec`."""

    def synthesize(
        self,
        *,
        spec_id: str,
        goal: str,
        tools: tuple[ToolSchema, ...],
        evaluator_id: str,
        criteria: str = "",
    ) -> AgentSpec: ...


def _render_tool(tool: ToolSchema) -> str:
    properties = tool.parameters.get("properties", {}) if isinstance(tool.parameters, dict) else {}
    required = set(tool.parameters.get("required", [])) if isinstance(tool.parameters, dict) else set()
    if properties:
        params = ", ".join(
            f"{name}{'' if name in required else '?'}: {spec.get('type', 'any') if isinstance(spec, dict) else 'any'}"
            for name, spec in properties.items()
        )
    else:
        params = ""
    description = f" -- {tool.description}" if tool.description else ""
    return f"- {tool.name}({params}){description}"


def compose_system_prompt(
    *, goal: str, tools: tuple[ToolSchema, ...], evaluator_id: str, criteria: str = ""
) -> str:
    """Assemble a starting system prompt. Domain content comes only from the arguments."""
    sections = [
        "You are an autonomous agent. Your goal, stated by the operator, is:",
        goal.strip(),
        "",
        "Available tools:",
    ]
    if tools:
        sections.extend(_render_tool(tool) for tool in tools)
    else:
        sections.append("(none -- answer directly)")
    sections += [
        "",
        "How to work:",
        "- At each step either call exactly one of the listed tools or give your final answer.",
        "- Never call a tool that is not listed above, and never invent its results.",
        "- Read each tool result before deciding the next step; if a call errors, "
        "fix the arguments rather than repeating it unchanged.",
        "- Stop as soon as you can support a complete answer, and not before.",
        "",
        f"Your final answer is judged by evaluator {evaluator_id!r}.",
    ]
    if criteria.strip():
        sections += ["It rewards:", criteria.strip()]
    return "\n".join(sections)


class TemplateSynthesizer:
    """Deterministic synthesis: no model call, fully reproducible.

    This is the default so the loop can run end to end offline. It produces a
    deliberately plain starting spec -- a competent baseline with obvious room
    to improve, which is what makes the subsequent mutations legible.
    """

    def __init__(
        self,
        *,
        strategy: OrchestrationStrategy = OrchestrationStrategy.REACT,
        max_steps: int = DEFAULT_MAX_STEPS,
        memory: MemoryConfig | None = None,
    ) -> None:
        self._strategy = strategy
        self._max_steps = max_steps
        self._memory = memory or MemoryConfig(kind=MemoryKind.FULL_TRANSCRIPT)

    def synthesize(
        self,
        *,
        spec_id: str,
        goal: str,
        tools: tuple[ToolSchema, ...],
        evaluator_id: str,
        criteria: str = "",
    ) -> AgentSpec:
        if not goal.strip():
            raise ValueError("goal must not be blank")
        return AgentSpec(
            spec_id=spec_id,
            system_prompt=compose_system_prompt(
                goal=goal, tools=tools, evaluator_id=evaluator_id, criteria=criteria
            ),
            tools=tuple(tool.name for tool in tools),
            strategy=self._strategy,
            memory=self._memory,
            stopping=StoppingConditions(max_steps=self._max_steps),
        )


class ModelSynthesizer:
    """Model-assisted synthesis: asks a :class:`TextGenerator` to write the prompt.

    Falls back to the template on any failure or empty completion, so a flaky
    model degrades the starting spec rather than breaking the run. The meta
    prompt describes the *shape* of a good agent prompt and injects the goal as
    data; it says nothing about any particular domain.
    """

    _META_SYSTEM = (
        "You write system prompts for tool-using autonomous agents. "
        "Given a goal and a tool list, output only the system prompt itself: "
        "no preamble, no commentary, no code fences. Be specific about method "
        "and about when to stop. Do not invent tools beyond those listed."
    )

    def __init__(self, generator: TextGenerator, *, fallback: SpecSynthesizer | None = None) -> None:
        self._generator = generator
        self._fallback = fallback or TemplateSynthesizer()

    def synthesize(
        self,
        *,
        spec_id: str,
        goal: str,
        tools: tuple[ToolSchema, ...],
        evaluator_id: str,
        criteria: str = "",
    ) -> AgentSpec:
        baseline = self._fallback.synthesize(
            spec_id=spec_id,
            goal=goal,
            tools=tools,
            evaluator_id=evaluator_id,
            criteria=criteria,
        )
        request = "\n".join(
            [
                "Goal:",
                goal.strip(),
                "",
                "Tools:",
                *(_render_tool(tool) for tool in tools),
                "",
                f"The final answer is judged by evaluator {evaluator_id!r}.",
                criteria,
                "",
                "Write the system prompt.",
            ]
        )
        try:
            written = self._generator.complete(self._META_SYSTEM, request)
        except Exception:
            return baseline
        if not written or not written.strip():
            return baseline
        return baseline.model_copy(update={"system_prompt": written.strip()})
