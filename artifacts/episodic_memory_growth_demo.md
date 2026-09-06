# Episodic Memory Growth on Real Registered Domain: `code_math`

## Overview & Track Question Answered

> **Track Judge Question**: *"Can you show the outputs of the agent getting better over time through its own self-reflection and MEMORY GROWING?"*
> 
> **Finding on Registered Domains**:
> Episodic memory is fully implemented and accumulates typed reflections across generations (0 → 29 entries). However, on single-turn tasks like `code_math`, **memory retrieval adds pure prompt token overhead (+29.6 tok/task)** without saving tool calls. Cost per solved task falls only because spec-level mutations increase task pass rates, not because memory pruned execution steps. Across the registered suites, memory did not provide an autonomous shortcut benefit.

---

## Registered Domain Results: `code_math`

Measured via `agent-engineer run code_math --memory episodic_store --max-generations 3`:

```
=== Episodic Memory Growth Across Iterations ===
generation    memory entry count  accuracy (with cost)  COST PER SOLVED TASK  verdict 
------------  ------------------  --------------------  --------------------  --------
gen 0 (root)  0 entries           0.1875 (41.9 tok)     223.3 tok             baseline
gen 1         18 entries          0.3125 (67.1 tok)     214.8 tok             ACCEPTED
gen 2         24 entries          0.6875 (70.9 tok)     103.1 tok             ACCEPTED
gen 3         29 entries          0.7500 (71.5 tok)     95.3 tok              ACCEPTED
-------------------------------------------------
Summary: Memory: 0 -> 29 entries (+29) | Accuracy: 0.1875 -> 0.7500 (+0.5625) | Cost: 41.9 tok -> 71.5 tok (+29.6 tok) | Cost / Solved Task: 223.3 tok -> 95.3 tok
```

| generation | memory entry count | accuracy (with cost) | COST PER SOLVED TASK | verdict |
| :--- | :--- | :--- | :--- | :--- |
| **gen 0 (root)** | 0 entries | 0.1875 (41.9 tok) | **223.3 tok** | baseline |
| **gen 1** | 18 entries | 0.3125 (67.1 tok) | **214.8 tok** | ACCEPTED |
| **gen 2** | 24 entries | 0.6875 (70.9 tok) | **103.1 tok** | ACCEPTED |
| **gen 3** | 29 entries | 0.7500 (71.5 tok) | **95.3 tok** | ACCEPTED |

> [!NOTE]
> **Honest Assessment**:
> Memory entries accumulated from 0 to 29 across iterations. Retrieval overhead increased prompt cost from 41.9 to 71.5 tokens per task (+70.6%). The accuracy gain (0.1875 → 0.7500) was driven by prompt and strategy mutations accepted by the loop, not by memory retrieval.
