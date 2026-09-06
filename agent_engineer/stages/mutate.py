"""Stage 4: emit exactly one targeted :class:`Mutation` for the dominant cause.

One mutation per generation, not a bundle. A bundle would make the measured
delta uninterpretable: if three edits ship together and the score moves, the
loop has learned nothing about which edit did it. Every mutation therefore
carries the :class:`Diagnosis` that motivated it and names the single cause it
aims at, and the next generation's delta is attributable to that edit alone.

Each cause maps to an ordered ladder of moves. The first applicable move this
lineage has not already tried is taken, so a cause that survives its first fix
escalates -- prompt guidance, then structural change -- instead of the loop
proposing the same edit forever.

The guidance text the prompt-rewrite moves append is about *agent behaviour*
(read tool errors, do not stop early, keep observations), which is true in every
domain. Domain-specific wording reaches a prompt only by flowing in from the
evaluator's own rationale, through the diagnosis -- never from a literal here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from agent_engineer.schemas import (
    AgentSpec,
    Diagnosis,
    FailureCause,
    MemoryConfig,
    MemoryKind,
    Mutation,
    MutationKind,
    OrchestrationStrategy,
)

MAX_STEP_CEILING = 64
DEFAULT_RETRIEVAL_K = 6


@dataclass(frozen=True)
class Proposal:
    """A candidate edit, before it is packaged as a :class:`Mutation`."""

    kind: MutationKind
    target_path: str
    before: str
    after: str
    rationale: str
    child: AgentSpec


Move = Callable[[AgentSpec, Diagnosis, str], "Proposal | None"]


@runtime_checkable
class Mutator(Protocol):
    """Proposes one mutation against a diagnosis, or None when it has no move left."""

    def propose_with_child(
        self,
        spec: AgentSpec,
        diagnosis: Diagnosis,
        *,
        mutation_id: str,
        child_spec_id: str,
        already_tried: frozenset[tuple[str, str, str]] = frozenset(),
    ) -> tuple[Mutation, AgentSpec] | None: ...


# --------------------------------------------------------------------------- #
# Prompt guidance, keyed by cause. Behavioural, never domain-specific.
# --------------------------------------------------------------------------- #

_GUIDANCE: dict[FailureCause, str] = {
    FailureCause.WRONG_TOOL_SELECTED: (
        "Before each step, name the one piece of information you still need, then pick the "
        "single listed tool that produces it. If no listed tool produces it, say so in your "
        "answer rather than guessing. Do not answer from assumption while a tool that would "
        "settle the question is still available."
    ),
    FailureCause.TOOL_MISUSE: (
        "Check every argument against the tool's parameter list before calling it, including "
        "types and required fields. If a call returns an error, read the error and change the "
        "arguments; never repeat the identical call, and never proceed as though an errored "
        "call had succeeded."
    ),
    FailureCause.FAULTY_REASONING: (
        "State your plan before acting, and after each tool result state what it rules in or "
        "out. Every claim in your final answer must trace to a specific tool result or to the "
        "task statement; if it traces to neither, drop it."
    ),
    FailureCause.CONTEXT_LOSS: (
        "Keep a running list of the facts you have established and the step each came from, "
        "and restate it before you answer. Never re-request information you already hold."
    ),
    FailureCause.PREMATURE_STOP: (
        "Before answering, check whether any part of the task is still unverified. If it is, "
        "keep working -- you have budget remaining. Answer only when every part of the request "
        "is covered."
    ),
    FailureCause.STOPPING_CONDITION_HIT: (
        "Work efficiently: pick the step with the highest information gain, avoid exploratory "
        "calls that do not narrow the problem, and do not re-fetch what you already have."
    ),
    FailureCause.OUTPUT_FORMAT_VIOLATION: (
        "Re-read the required form of the answer before you emit it, and emit exactly that "
        "form -- no preamble, no commentary, no extra fields. The content being right does not "
        "excuse the shape being wrong."
    ),
}

_STRATEGY_LADDER: tuple[OrchestrationStrategy, ...] = (
    OrchestrationStrategy.SINGLE_SHOT,
    OrchestrationStrategy.REACT,
    OrchestrationStrategy.PLAN_THEN_EXECUTE,
    OrchestrationStrategy.REFLEXION,
    OrchestrationStrategy.TREE_SEARCH,
)


def _evidence_digest(diagnosis: Diagnosis, cause: FailureCause, limit: int = 3) -> str:
    """The strongest evidence lines for one cause, quoted back into the prompt."""
    lines = [
        attribution.evidence
        for attribution in sorted(diagnosis.attributions, key=lambda a: -a.confidence)
        if attribution.cause is cause
    ][:limit]
    return "\n".join(f"- {line}" for line in lines)


def _child(spec: AgentSpec, child_spec_id: str, **updates: object) -> AgentSpec:
    """Apply an edit and re-validate.

    ``model_copy`` skips validation, so the result is round-tripped through
    ``model_validate``: a mutation that would produce an illegal spec -- a
    stop-on-tool that no longer exists, say -- fails here rather than reaching
    the runner.
    """
    edited = spec.model_copy(
        update={"spec_id": child_spec_id, "parent_spec_id": spec.spec_id, **updates}
    )
    return AgentSpec.model_validate(edited.model_dump())


# --------------------------------------------------------------------------- #
# Moves
# --------------------------------------------------------------------------- #


def _move_prompt_guidance(
    spec: AgentSpec, diagnosis: Diagnosis, child_spec_id: str
) -> Proposal | None:
    cause = diagnosis.dominant_cause
    if cause is None:
        return None
    guidance = _GUIDANCE[cause]
    if guidance in spec.system_prompt:
        return None
    evidence = _evidence_digest(diagnosis, cause)
    block = [f"Correction, from observed failures attributed to {cause.value}:", guidance]
    if evidence:
        block += ["What went wrong on the last run:", evidence]
    after = spec.system_prompt.rstrip() + "\n\n" + "\n".join(block) + "\n"
    return Proposal(
        kind=MutationKind.SYSTEM_PROMPT_REWRITE,
        target_path="system_prompt",
        before=spec.system_prompt,
        after=after,
        rationale=(
            f"The dominant cause is {cause.value}. Adding explicit behavioural guidance for it "
            "is the smallest edit that can change the agent's conduct at the step where it fails."
        ),
        child=_child(spec, child_spec_id, system_prompt=after),
    )


def _move_raise_step_budget(
    spec: AgentSpec, diagnosis: Diagnosis, child_spec_id: str
) -> Proposal | None:
    current = spec.stopping.max_steps
    if current is None or current >= MAX_STEP_CEILING:
        return None
    raised = min(MAX_STEP_CEILING, current * 2)
    stopping = spec.stopping.model_copy(update={"max_steps": raised})
    return Proposal(
        kind=MutationKind.STOPPING_ADJUSTED,
        target_path="stopping.max_steps",
        before=str(current),
        after=str(raised),
        rationale=(
            f"Runs are being cut off at {current} steps before finishing, so the budget is "
            "binding on the outcome rather than the agent's competence. Doubling it separates "
            "the two."
        ),
        child=_child(spec, child_spec_id, stopping=stopping),
    )


def _move_escalate_strategy(
    spec: AgentSpec, diagnosis: Diagnosis, child_spec_id: str
) -> Proposal | None:
    if spec.strategy not in _STRATEGY_LADDER:
        return None
    index = _STRATEGY_LADDER.index(spec.strategy)
    if index + 1 >= len(_STRATEGY_LADDER):
        return None
    upgraded = _STRATEGY_LADDER[index + 1]
    cause = diagnosis.dominant_cause or FailureCause.FAULTY_REASONING
    return Proposal(
        kind=MutationKind.STRATEGY_CHANGED,
        target_path="strategy",
        before=spec.strategy.value,
        after=upgraded.value,
        rationale=(
            f"Prompt-level guidance did not fix {cause.value}, so the loop shape is the "
            f"constraint: {upgraded.value} gives the agent an explicit step for the "
            "deliberation it is currently skipping."
        ),
        child=_child(spec, child_spec_id, strategy=upgraded),
    )


def _move_retain_full_context(
    spec: AgentSpec, diagnosis: Diagnosis, child_spec_id: str
) -> Proposal | None:
    if spec.memory.kind is MemoryKind.FULL_TRANSCRIPT:
        return None
    memory = MemoryConfig(
        kind=MemoryKind.FULL_TRANSCRIPT,
        max_tokens=spec.memory.max_tokens,
        persist_across_runs=spec.memory.persist_across_runs,
    )
    return Proposal(
        kind=MutationKind.MEMORY_RECONFIGURED,
        target_path="memory.kind",
        before=spec.memory.kind.value,
        after=memory.kind.value,
        rationale=(
            "Information established early is not reaching the step that needs it under "
            f"{spec.memory.kind.value}. Retaining the full transcript removes the lossy hop."
        ),
        child=_child(spec, child_spec_id, memory=memory),
    )


def _move_add_episodic_store(
    spec: AgentSpec, diagnosis: Diagnosis, child_spec_id: str
) -> Proposal | None:
    if spec.memory.kind is MemoryKind.EPISODIC_STORE and spec.memory.persist_across_runs:
        return None
    memory = MemoryConfig(
        kind=MemoryKind.EPISODIC_STORE,
        max_tokens=spec.memory.max_tokens,
        retrieval_k=spec.memory.retrieval_k or DEFAULT_RETRIEVAL_K,
        persist_across_runs=True,
    )
    return Proposal(
        kind=MutationKind.MEMORY_RECONFIGURED,
        target_path="memory.kind",
        before=spec.memory.kind.value,
        after=memory.kind.value,
        rationale=(
            "Prior run experience and failure lessons were lost between iterations. Configuring "
            f"an episodic store with retrieval_k={memory.retrieval_k} and persist_across_runs=True "
            "allows the agent to accumulate and retrieve distilled memory across runs."
        ),
        child=_child(spec, child_spec_id, memory=memory),
    )


def _move_add_retrieval(
    spec: AgentSpec, diagnosis: Diagnosis, child_spec_id: str
) -> Proposal | None:
    if spec.memory.kind is MemoryKind.VECTOR_RETRIEVAL:
        return None
    memory = MemoryConfig(
        kind=MemoryKind.VECTOR_RETRIEVAL,
        max_tokens=spec.memory.max_tokens,
        retrieval_k=DEFAULT_RETRIEVAL_K,
        persist_across_runs=spec.memory.persist_across_runs,
    )
    return Proposal(
        kind=MutationKind.MEMORY_RECONFIGURED,
        target_path="memory.kind",
        before=spec.memory.kind.value,
        after=memory.kind.value,
        rationale=(
            "Full retention did not stop the context loss, so the problem is finding the "
            f"relevant earlier fact, not storing it. Retrieval over {DEFAULT_RETRIEVAL_K} "
            "relevant items targets that directly."
        ),
        child=_child(spec, child_spec_id, memory=memory),
    )


def _move_require_final_answer(
    spec: AgentSpec, diagnosis: Diagnosis, child_spec_id: str
) -> Proposal | None:
    if spec.stopping.require_final_answer:
        return None
    stopping = spec.stopping.model_copy(update={"require_final_answer": True})
    return Proposal(
        kind=MutationKind.STOPPING_ADJUSTED,
        target_path="stopping.require_final_answer",
        before="false",
        after="true",
        rationale=(
            "Runs are ending without an answer at all; require one before a run counts as done."
        ),
        child=_child(spec, child_spec_id, stopping=stopping),
    )


def _move_reorder_tools(
    spec: AgentSpec, diagnosis: Diagnosis, child_spec_id: str
) -> Proposal | None:
    if len(spec.tools) < 2:
        return None
    reordered = tuple(sorted(spec.tools))
    if reordered == spec.tools:
        reordered = spec.tools[-1:] + spec.tools[:-1]
    if reordered == spec.tools:
        return None
    return Proposal(
        kind=MutationKind.TOOL_SET_REORDERED,
        target_path="tools",
        before=", ".join(spec.tools),
        after=", ".join(reordered),
        rationale=(
            "The agent keeps reaching for the wrong tool; tool order is the cheapest lever on "
            "selection, so reorder the list and measure whether presentation order is carrying "
            "any of the failure."
        ),
        child=_child(spec, child_spec_id, tools=reordered),
    )


LADDERS: dict[FailureCause, tuple[Move, ...]] = {
    # Reaching for the wrong tool, or none at all: presentation order first
    # (cheapest lever on selection), then the loop shape.
    FailureCause.WRONG_TOOL_SELECTED: (
        _move_prompt_guidance,
        _move_reorder_tools,
        _move_escalate_strategy,
    ),
    # Calling the right tool wrong: behavioural guidance on argument-checking
    # first, then a loop shape that gives the model a deliberate step before
    # each call. There is deliberately no third move here that is not
    # actually about how a tool is called -- a step-budget raise or a memory
    # change does not address malformed arguments, and proposing one anyway
    # is exactly the diagnosis-mutation mismatch this ladder used to have.
    FailureCause.TOOL_MISUSE: (
        _move_prompt_guidance,
        _move_escalate_strategy,
    ),
    FailureCause.FAULTY_REASONING: (
        _move_prompt_guidance,
        _move_escalate_strategy,
        _move_retain_full_context,
    ),
    FailureCause.CONTEXT_LOSS: (
        _move_retain_full_context,
        _move_prompt_guidance,
        _move_add_episodic_store,
        _move_add_retrieval,
    ),
    FailureCause.PREMATURE_STOP: (
        _move_prompt_guidance,
        _move_require_final_answer,
        _move_escalate_strategy,
    ),
    FailureCause.STOPPING_CONDITION_HIT: (
        _move_raise_step_budget,
        _move_prompt_guidance,
        _move_retain_full_context,
    ),
    # Output shape, not content: guidance on the required form, then a loop
    # shape (REFLEXION) that gives the model an explicit self-check-and-revise
    # step before the answer is final -- a genuine retry-on-format-failure,
    # not a tool-order or step-budget move that has nothing to do with shape.
    FailureCause.OUTPUT_FORMAT_VIOLATION: (
        _move_prompt_guidance,
        _move_escalate_strategy,
    ),
}
"""Per-cause escalation ladders, cheapest and most targeted move first.

Every move on every ladder must be one whose rationale is actually true of
the cause it is listed under. A move that is only honest for a different
cause does not belong here, even as a last resort -- see the module
docstring and the `_FALLBACK_LADDER` note below.
"""

_FALLBACK_LADDER: tuple[Move, ...] = (
    _move_prompt_guidance,
    _move_escalate_strategy,
)
"""Tried after a cause's own ladder is exhausted.

Deliberately short, and deliberately cause-agnostic in truth, not just in
applicability: prompt guidance and a loop-shape escalation are legitimate
responses to *any* behavioural cause. The moves that used to live here --
raising the step budget, changing memory, reordering tools -- each carry a
rationale that asserts something specific to a *different* cause (a budget
cut-off, context loss, wrong-tool selection). Firing one of them for a cause
it was never diagnosed for is not a fallback, it is the loop making an
unrelated edit and reporting it as reasoned. When both moves here are
exhausted the mutator returns ``None`` and the lineage ends honestly, which
is the documented behaviour an exhausted ladder is supposed to produce.
"""


class LadderMutator:
    """Emits one mutation per generation, escalating when a cause survives its fix."""

    def __init__(self, ladders: dict[FailureCause, tuple[Move, ...]] | None = None) -> None:
        self._ladders = ladders or LADDERS

    def propose_with_child(
        self,
        spec: AgentSpec,
        diagnosis: Diagnosis,
        *,
        mutation_id: str,
        child_spec_id: str,
        already_tried: frozenset[tuple[str, str, str]] = frozenset(),
    ) -> tuple[Mutation, AgentSpec] | None:
        """Emit one mutation aimed at the dominant cause, plus the spec it produces.

        Returns None when nothing failed, or when every move on the ladder has
        already been tried on this lineage -- an exhausted ladder ends the run
        honestly instead of re-proposing an edit that has already been measured.
        """
        cause = diagnosis.dominant_cause
        if cause is None:
            return None  # nothing failed; there is nothing to aim at

        ordered: list[Move] = list(self._ladders.get(cause, ()))
        ordered += [move for move in _FALLBACK_LADDER if move not in ordered]

        for move in ordered:
            proposal = move(spec, diagnosis, child_spec_id)
            if proposal is None or proposal.before == proposal.after:
                continue
            if (proposal.kind.value, proposal.target_path, proposal.after) in already_tried:
                continue
            mutation = Mutation(
                mutation_id=mutation_id,
                kind=proposal.kind,
                target_path=proposal.target_path,
                before=proposal.before,
                after=proposal.after,
                rationale=proposal.rationale,
                motivating_diagnosis=diagnosis,
                motivating_cause=cause,
                parent_spec_id=spec.spec_id,
                child_spec_id=proposal.child.spec_id,
            )
            return mutation, proposal.child
        return None

    def propose(
        self,
        spec: AgentSpec,
        diagnosis: Diagnosis,
        *,
        mutation_id: str,
        child_spec_id: str,
        already_tried: frozenset[tuple[str, str, str]] = frozenset(),
    ) -> Mutation | None:
        """As :meth:`propose_with_child`, discarding the child spec."""
        proposed = self.propose_with_child(
            spec,
            diagnosis,
            mutation_id=mutation_id,
            child_spec_id=child_spec_id,
            already_tried=already_tried,
        )
        return proposed[0] if proposed else None
