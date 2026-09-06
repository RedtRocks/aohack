# Episodic Memory Loop: Self-Reflection & Memory Growth Across Iterations

## Overview & Track Question Answered

> **Track Judge Question**: *"Can you show the outputs of the agent getting better over time through its own self-reflection and MEMORY GROWING?"*
> 
> **Answer**: **Yes.** The agent accumulates typed episodic memories across iterations. Successful trajectories and failure self-reflections (diagnosed causes, evidence, and tool sequencing lessons) are stored in an explicit, resettable `EpisodicMemoryStore`. At run time, top-$k$ relevant entries are retrieved into the agent's context.

---

## 1. Primary Finding: Multi-Turn Domain with Shared Subgoals (`multi_step_orchestration`)

*Core Finding: Even with **flat accuracy** (0/3 mutations accepted, all REVERTED, exactly matching live model run behavior), memory accumulation measurably cuts the **COST PER SOLVED TASK** by eliminating blind exploratory tool discovery.*

```
=== Episodic Memory Growth Across Iterations ===
generation    memory entry count  accuracy (with cost)  COST PER SOLVED TASK  verdict 
------------  ------------------  --------------------  --------------------  --------
gen 0 (root)  0 entries           0.8333 (203.0 tok)    243.6 tok             baseline
gen 1         9 entries           0.8333 (152.5 tok)    183.0 tok             REVERTED
gen 2         9 entries           0.8333 (152.5 tok)    183.0 tok             REVERTED
gen 3         10 entries          0.8333 (152.5 tok)    183.0 tok             REVERTED
-------------------------------------------------
Summary: Memory: 0 -> 10 entries (+10) | Accuracy: 0.8333 -> 0.8333 (+0.0000) | Cost: 203.0 tok -> 152.5 tok (-50.5 tok) | Cost / Solved Task: 243.6 tok -> 183.0 tok
```

| generation | memory entry count | accuracy (with cost) | COST PER SOLVED TASK |
| :--- | :--- | :--- | :--- |
| **gen 0 (root)** | 0 entries | 0.8333 (203.0 tok) | **243.6 tok** |
| **gen 1** | 9 entries | 0.8333 (152.5 tok) | **183.0 tok** (-60.6 tok) |
| **gen 2** | 9 entries | 0.8333 (152.5 tok) | **183.0 tok** (-60.6 tok) |
| **gen 3** | 10 entries | 0.8333 (152.5 tok) | **183.0 tok** (-60.6 tok) |

> [!IMPORTANT]
> **Why this matters**:
> Across generations, the loop accepted **nothing** (all mutations reverted). Yet, the agent's memory store grew from **0 → 10 entries**, and the **cost per solved task fell from 243.6 tok to 183.0 tok (-24.9%)**. The agent is reusing what it learned through post-diagnosis reflection instead of rediscovering auth and routing tokens every run.

---

## 2. Contrast: Single-Shot Domain (`code_math`)

*Honest Negative Finding on Single-Turn Tasks: On tasks with zero tool calls, memory retrieval is pure token overhead (+32.1 tokens/task); it only amortizes if accuracy rises.*

```
=== Episodic Memory Growth Across Iterations ===
generation    memory entry count  accuracy (with cost)  COST PER SOLVED TASK  verdict 
------------  ------------------  --------------------  --------------------  --------
gen 0 (root)  0 entries           0.1875 (41.9 tok)     223.3 tok             baseline
gen 1         25 entries          0.7500 (71.5 tok)     95.3 tok              ACCEPTED
gen 2         27 entries          0.8750 (72.8 tok)     83.1 tok              ACCEPTED
gen 3         29 entries          1.0000 (74.0 tok)     74.0 tok              ACCEPTED
-------------------------------------------------
Summary: Memory: 0 -> 29 entries (+29) | Accuracy: 0.1875 -> 1.0000 (+0.8125) | Cost: 41.9 tok -> 74.0 tok (+32.1 tok) | Cost / Solved Task: 223.3 tok -> 74.0 tok
```

| generation | memory entry count | accuracy (with cost) | COST PER SOLVED TASK |
| :--- | :--- | :--- | :--- |
| **gen 0 (root)** | 0 entries | 0.1875 (41.9 tok) | **223.3 tok** |
| **gen 1** | 25 entries | 0.7500 (71.5 tok) | **95.3 tok** |
| **gen 2** | 27 entries | 0.8750 (72.8 tok) | **83.1 tok** |
| **gen 3** | 29 entries | 1.0000 (74.0 tok) | **74.0 tok** |

> [!NOTE]
> **Key Comparison**:
> *"Memory on single-turn tasks is pure overhead (+32.1 tokens of retrieval, 0 tool calls saved); on multi-turn tasks with shared subgoals, cost per solved task falls from 243.6 tok to 183.0 tok."*
