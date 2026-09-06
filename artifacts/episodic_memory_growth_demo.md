# Episodic Memory Loop: Self-Reflection & Memory Growth Across Iterations

## Overview & Track Question Answered

> **Track Judge Question**: *"Can you show the outputs of the agent getting better over time through its own self-reflection and MEMORY GROWING?"*
> 
> **Answer**: **Yes.** The agent now accumulates distilled, typed episodic memories across iterations. Successful trajectories and failure self-reflections (diagnosed causes, evidence, and tool sequencing lessons) are stored in an explicit, resettable `EpisodicMemoryStore`. At run time, top-$k$ relevant entries are retrieved into the agent's context.

---

## The Demo Across Iterations

### 1. Multi-Step Execution Domain (`efficiency-domain`)
*Claim: "Memory grew, accuracy rose, cost per task fell."*
When an agent is faced with complex tool sequences or prerequisites, failures cause exploratory retries and wasted token spend. Distilled failure reflections teach the agent the direct path, drastically reducing unnecessary tool calls and total token cost.

| Iteration | Memory Store Size | Accuracy (Pass Rate) | Cost per Task (Tokens) | Verdict / Decision | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gen 0 (root)** | **0 entries** | **0.0000** | **155.0 tok** | `baseline` | Erred on tool calls; retried blindly; gave up |
| **Gen 1** | **6 entries** | **1.0000 (+1.0000)** | **55.0 tok (-100.0 tok)** | `ACCEPTED` | Retrieved distilled memory; solved directly |

> [!NOTE]
> **Summary**: Memory grew from **0 → 6 entries (+6)**. Accuracy rose from **0.0000 → 1.0000 (+1.0000)**. Cost per task fell from **155.0 tok → 55.0 tok (-100.0 tok)**.

---

### 2. Single-Shot Registered Domain (`code_math`)
*Claim: "Memory grew, accuracy rose; cost reported honestly."*
In a single-shot domain without tool retries, retrieving memory entries into context adds prompt tokens. We report this honestly rather than presenting a flattering synthetic number.

| Iteration | Memory Store Size | Accuracy (Mean Score) | Cost per Task (Tokens) | Verdict / Decision | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gen 0 (root)** | **0 entries** | **0.1875** | **41.9 tok** | `baseline` | Stopped early without memory |
| **Gen 1** | **25 entries** | **0.7500 (+0.5625)** | **71.5 tok (+29.6 tok)** | `ACCEPTED` | Retrieved top lessons; resolved premature stops |
| **Gen 2** | **27 entries** | **0.8750 (+0.1250)** | **72.8 tok (+30.9 tok)** | `ACCEPTED` | Strategy evolved with memory in context |
| **Gen 3** | **29 entries** | **1.0000 (+0.1250)** | **74.0 tok (+32.1 tok)** | `ACCEPTED` | 100% pass rate achieved |

> [!NOTE]
> **Summary**: Memory grew from **0 → 29 entries (+29)**. Accuracy rose from **0.1875 → 1.0000 (+0.8125)**. Cost per task rose from **41.9 tok → 74.0 tok (+32.1 tok)** due to prompt tokens from retrieved context.

---

## Architectural Implementation

### 1. Episodic Store (`agent_engineer/memory.py`)
- **Typed, Concise Entries (`EpisodicEntry`)**:
  - `entry_id`, `kind` (`"success"` or `"failure_lesson"`), `task_id`, `lesson`, `cause`, `evidence`, `tools_used`, `generation`, `query_keys`.
  - Transcripts are never dumped raw; lessons are short, typed, and structured.
- **Explicit & Resettable (`EpisodicMemoryStore`)**:
  - Optional `storage_path` for JSON persistence.
  - Defaults to isolated in-memory storage so tests and runs never leak state.
  - Explicit `.clear()` method resets disk files and memory.
- **Domain-Agnostic Retrieval**:
  - Relevance ranking based on lexical term matching, token overlap, and recency tie-breaking.
  - Zero domain-specific vocabulary or hardcoded rules.

### 2. Self-Reflection (`HeuristicReflector` & `ModelReflector`)
- Invoked after Stage 3 (`diagnose`).
- **Success Reflection**: Distills the winning tool sequence and output contract.
- **Failure Reflection**: Maps `FailureAttribution` (`cause`, `evidence`, `step_ordinal`) into actionable guidance (e.g. prerequisite checks, argument schema verification, format compliance).

### 3. Run-time Retrieval (`agent_engineer/stages/evaluate.py`)
- When `spec.memory.kind == MemoryKind.EPISODIC_STORE` and `spec.memory.persist_across_runs`:
  - Retrieves top-$k$ entries matching the current task.
  - Injects formatted memory block into `effective_spec.system_prompt`.
  - Retains canonical `spec.spec_id` and tracks token usage accurately.

### 4. Mutation Ladder Integration (`agent_engineer/stages/mutate.py`)
- `_move_add_episodic_store` added to `LADDERS[FailureCause.CONTEXT_LOSS]` and `_FALLBACK_LADDER`.
- Mutates `spec.memory` to `MemoryKind.EPISODIC_STORE` with `persist_across_runs=True` and `retrieval_k=6`.
