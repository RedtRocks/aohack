"""Episodic memory store, reflection distillation, and cross-iteration retrieval.

This module owns episodic memory persistence across iterations. It is completely
domain-agnostic: entries store structural lessons, tool call sequences, and failure
attributions, without domain-specific vocabulary or logic.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from agent_engineer.ports import TextGenerator
from agent_engineer.schemas import (
    AgentSpec,
    Diagnosis,
    FailureAttribution,
    FailureCause,
    Trajectory,
)

if TYPE_CHECKING:
    from agent_engineer.stages.evaluate import EvaluationRun, RunRecord

__all__ = [
    "EpisodicEntry",
    "EpisodicMemoryStore",
    "Reflector",
    "HeuristicReflector",
    "ModelReflector",
    "distill_reflections",
]

_WORD_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    """Extract lowercased word tokens from text."""
    return [w.lower() for w in _WORD_PATTERN.findall(text)]


class EpisodicEntry(BaseModel):
    """A short, typed memory entry distilled from an agent's execution episode."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entry_id: str = Field(min_length=1)
    kind: Literal["success", "failure_lesson"]
    task_id: str = Field(min_length=1)
    lesson: str = Field(min_length=1)
    cause: FailureCause | None = None
    evidence: str = ""
    tools_used: tuple[str, ...] = ()
    generation: int = 0
    query_keys: tuple[str, ...] = ()
    created_at: float = Field(default_factory=time.time)

    def summary_line(self) -> str:
        """Render a single concise line for prompt injection or logging."""
        if self.kind == "success":
            return f"[SUCCESS] Task {self.task_id}: {self.lesson}"
        cause_str = self.cause.value if self.cause else "general"
        return f"[FAILURE LESSON] Cause: {cause_str} | Task {self.task_id}: {self.lesson}"


class EpisodicMemoryStore:
    """An episodic store that accumulates and retrieves lessons across iterations.

    Persistence is explicit and resettable. If storage_path is provided, entries
    are synced to disk in JSON format. If storage_path is None, the store operates
    in-memory only, preventing cross-test state leakage.
    """

    def __init__(self, storage_path: Path | str | None = None) -> None:
        self._path = Path(storage_path) if storage_path is not None else None
        self._entries: list[EpisodicEntry] = []
        if self._path is not None and self._path.is_file():
            self.load()

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> tuple[EpisodicEntry, ...]:
        return tuple(self._entries)

    def add(self, entry: EpisodicEntry) -> None:
        """Add a single entry and persist if configured."""
        # Avoid duplicate identical lessons for the same task
        for existing in self._entries:
            if existing.entry_id == entry.entry_id or (
                existing.task_id == entry.task_id
                and existing.kind == entry.kind
                and existing.lesson == entry.lesson
            ):
                return
        self._entries.append(entry)
        if self._path is not None:
            self.save()

    def add_many(self, entries: Iterable[EpisodicEntry]) -> None:
        """Add multiple entries."""
        for entry in entries:
            self.add(entry)

    def clear(self) -> None:
        """Reset the store in-memory and on disk."""
        self._entries.clear()
        if self._path is not None and self._path.exists():
            self._path.unlink(missing_ok=True)

    def retrieve(self, query: str, k: int = 3) -> tuple[EpisodicEntry, ...]:
        """Retrieve the top-k relevant entries for a query string.

        Uses domain-agnostic lexical term matching with recency tie-breaking.
        """
        if not self._entries or k <= 0:
            return ()

        query_tokens = Counter(_tokenize(query))
        if not query_tokens:
            # Return most recent entries if query is empty
            return tuple(sorted(self._entries, key=lambda e: (-e.generation, -e.created_at))[:k])

        scored: list[tuple[float, float, int, EpisodicEntry]] = []
        total_docs = len(self._entries)

        for entry in self._entries:
            doc_tokens = Counter(
                _tokenize(
                    f"{entry.task_id} {entry.lesson} {entry.evidence} "
                    f"{' '.join(entry.tools_used)} {' '.join(entry.query_keys)}"
                )
            )
            # Compute term overlap score
            score = 0.0
            for q_tok, q_count in query_tokens.items():
                if q_tok in doc_tokens:
                    doc_count = doc_tokens[q_tok]
                    # Log-frequency weighting
                    tf = 1.0 + math.log(doc_count)
                    score += tf * (1.0 + math.log(q_count))

            # Prefer entries with matching task_id or tool names
            task_tokens = set(_tokenize(entry.task_id))
            if any(t in query_tokens for t in task_tokens):
                score += 3.0

            for tool in entry.tools_used:
                if tool.lower() in query_tokens:
                    score += 2.0

            scored.append((score, entry.created_at, entry.generation, entry))

        # Sort primarily by relevance score, then generation, then recency
        scored.sort(key=lambda item: (-item[0], -item[2], -item[1]))

        # Take top k, filtering out zero-score entries only if some scored > 0
        positive_scored = [item[3] for item in scored if item[0] > 0.0]
        if positive_scored:
            return tuple(positive_scored[:k])
        # If no positive matches, return the most recent entries
        return tuple(item[3] for item in scored[:k])

    def format_for_context(self, entries: Sequence[EpisodicEntry]) -> str:
        """Render retrieved entries into a prompt section for agent context."""
        if not entries:
            return ""
        lines = [
            "=== Episodic Memory (Lessons and trajectories from prior iterations) ===",
            "Use these relevant prior lessons to inform tool selection and avoid known pitfalls:",
        ]
        for entry in entries:
            lines.append(f"- {entry.summary_line()}")
        lines.append("=========================================================================")
        return "\n".join(lines)

    def save(self) -> None:
        """Serialize entries to storage_path."""
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [entry.model_dump(mode="json") for entry in self._entries]
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self) -> None:
        """Deserialize entries from storage_path."""
        if self._path is None or not self._path.is_file():
            return
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        self._entries = [EpisodicEntry.model_validate(item) for item in raw]


# --------------------------------------------------------------------------- #
# Reflection & Distillation
# --------------------------------------------------------------------------- #


@runtime_checkable
class Reflector(Protocol):
    """Distills execution trajectories and diagnoses into typed EpisodicEntry items."""

    def distill(
        self,
        spec: AgentSpec,
        run: EvaluationRun,
        diagnosis: Diagnosis | None = None,
        *,
        generation: int = 0,
    ) -> tuple[EpisodicEntry, ...]: ...


class HeuristicReflector:
    """Domain-agnostic distillation of trajectories into typed episodic lessons."""

    def distill(
        self,
        spec: AgentSpec,
        run: EvaluationRun,
        diagnosis: Diagnosis | None = None,
        *,
        generation: int = 0,
    ) -> tuple[EpisodicEntry, ...]:
        entries: list[EpisodicEntry] = []
        attr_by_traj: dict[str, FailureAttribution] = {}
        if diagnosis is not None:
            for attr in diagnosis.attributions:
                attr_by_traj[attr.trajectory_id] = attr

        for record in run.records:
            traj = record.trajectory
            task_id = traj.task_id
            traj_id = traj.trajectory_id

            if traj.verdict is not None and traj.verdict.passed:
                # 1. Success trajectory distillation
                tool_names = tuple(c.tool_name for c in traj.tool_calls)
                if tool_names:
                    tools_str = " -> ".join(tool_names)
                    lesson = f"Solved via tool sequence [{tools_str}]."
                elif traj.final_answer:
                    lesson = "Solved directly without external tool invocations."
                else:
                    lesson = "Completed successfully."

                query_keys = tuple(set(_tokenize(f"{task_id} {' '.join(tool_names)}")))
                entry = EpisodicEntry(
                    entry_id=f"{traj_id}::success",
                    kind="success",
                    task_id=task_id,
                    lesson=lesson,
                    tools_used=tool_names,
                    generation=generation,
                    query_keys=query_keys,
                )
                entries.append(entry)

            elif traj.failed:
                # 2. Failure self-reflection distillation
                attr = attr_by_traj.get(traj_id)
                cause = attr.cause if attr else FailureCause.FAULTY_REASONING
                evidence = attr.evidence if attr else "Run failed without specific attribution."
                step_ordinal = attr.step_ordinal if attr else None

                tool_names = tuple(c.tool_name for c in traj.tool_calls)
                lesson = self._distill_failure_lesson(traj, cause, evidence, step_ordinal)
                query_keys = tuple(
                    set(_tokenize(f"{task_id} {cause.value} {' '.join(tool_names)} {evidence}"))
                )

                entry = EpisodicEntry(
                    entry_id=f"{traj_id}::failure",
                    kind="failure_lesson",
                    task_id=task_id,
                    lesson=lesson,
                    cause=cause,
                    evidence=evidence,
                    tools_used=tool_names,
                    generation=generation,
                    query_keys=query_keys,
                )
                entries.append(entry)

        return tuple(entries)

    def _distill_failure_lesson(
        self,
        traj: Trajectory,
        cause: FailureCause,
        evidence: str,
        step_ordinal: int | None,
    ) -> str:
        """Produce a short, typed takeaway from failure evidence."""
        calls = traj.tool_calls

        if step_ordinal is not None and 0 <= step_ordinal < len(calls):
            failed_call = calls[step_ordinal]
            tool_name = failed_call.tool_name
            error_msg = failed_call.error or evidence
            if step_ordinal > 0:
                prior_tool = calls[step_ordinal - 1].tool_name
                return (
                    f"Calling tool '{tool_name}' after '{prior_tool}' failed: {error_msg}. "
                    f"Verify prerequisite data from earlier steps before invoking '{tool_name}'."
                )
            return (
                f"Calling tool '{tool_name}' failed: {error_msg}. "
                "Check argument structure and parameter schema."
            )

        if cause is FailureCause.TOOL_MISUSE:
            return (
                f"Tool misuse: {evidence}. "
                "Verify required arguments and validate return types before proceeding."
            )
        if cause is FailureCause.WRONG_TOOL_SELECTED:
            return (
                f"Tool selection error: {evidence}. "
                "Select tools whose interfaces directly match the required data."
            )
        if cause is FailureCause.CONTEXT_LOSS:
            return (
                f"Context loss: {evidence}. "
                "Carry earlier observations forward; avoid redundant or repeating queries."
            )
        if cause is FailureCause.STOPPING_CONDITION_HIT:
            return (
                f"Budget exhausted: {evidence}. "
                "Streamline plan to invoke high-information tools first."
            )
        if cause is FailureCause.PREMATURE_STOP:
            return (
                f"Premature termination: {evidence}. "
                "Continue executing steps while budget remains until the answer is validated."
            )
        if cause is FailureCause.OUTPUT_FORMAT_VIOLATION:
            return (
                f"Format violation: {evidence}. "
                "Format answer precisely to requested specification without commentary."
            )

        return (
            f"Reasoning breakdown: {evidence}. "
            "Ensure every conclusion is grounded in validated tool observations."
        )


class ModelReflector:
    """Model-assisted reflection using a TextGenerator, with HeuristicReflector fallback."""

    _SYSTEM_PROMPT = (
        "You are an expert self-reflection module for autonomous agents. "
        "Given a trajectory and diagnosis of why it failed or succeeded, produce "
        "exactly one short, actionable lesson (1-2 sentences) explaining how to "
        "succeed or avoid this error in future iterations. Reply with the lesson only."
    )

    def __init__(
        self, generator: TextGenerator, *, fallback: HeuristicReflector | None = None
    ) -> None:
        self._generator = generator
        self._fallback = fallback or HeuristicReflector()

    def distill(
        self,
        spec: AgentSpec,
        run: EvaluationRun,
        diagnosis: Diagnosis | None = None,
        *,
        generation: int = 0,
    ) -> tuple[EpisodicEntry, ...]:
        baseline_entries = self._fallback.distill(spec, run, diagnosis, generation=generation)
        refined_entries: list[EpisodicEntry] = []

        for entry in baseline_entries:
            if entry.kind != "failure_lesson":
                refined_entries.append(entry)
                continue
            prompt = (
                f"Task ID: {entry.task_id}\n"
                f"Cause: {entry.cause.value if entry.cause else 'none'}\n"
                f"Evidence: {entry.evidence}\n"
                f"Baseline lesson: {entry.lesson}\n"
                f"Tools: {', '.join(entry.tools_used) or 'none'}"
            )
            try:
                model_lesson = self._generator.complete(self._SYSTEM_PROMPT, prompt)
                model_lesson = model_lesson.strip().replace("\n", " ")
                if len(model_lesson) > 10:
                    entry = entry.model_copy(update={"lesson": model_lesson})
            except Exception:
                pass  # Keep baseline on any model error
            refined_entries.append(entry)

        return tuple(refined_entries)


def distill_reflections(
    spec: AgentSpec,
    run: EvaluationRun,
    diagnosis: Diagnosis | None = None,
    *,
    generation: int = 0,
    reflector: Reflector | None = None,
) -> tuple[EpisodicEntry, ...]:
    """Convenience helper to distill reflections from an evaluation run."""
    reflector = reflector or HeuristicReflector()
    return reflector.distill(spec, run, diagnosis, generation=generation)
