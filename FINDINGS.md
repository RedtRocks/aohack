# Empirical Findings

This document reports what the automated agent-engineer optimization loop measured, caught, and verified across four registered domains (`code_math`, `api_orchestration`, `extraction`, `mcp_everything`) and 378 live model API calls.

Every number in this document traces directly to an artifact in the repository.

---

## 1. Domain-Agnostic Diagnosis, Shown Live

The diagnosis engine (`agent_engineer/stages/diagnose.py`) attributed failure causes from execution trajectory structure alone, without importing any domain package or inspecting domain-specific vocabulary.

Across live runs on real models (TensorMux GLM-4-7-flash primary with OpenAI GPT-5-nano fallback; source: [`artifacts/cross_domain_live_comparison.md`](artifacts/cross_domain_live_comparison.md)):

* **`extraction` (14 tasks)**: Diagnosed dominant cause as `output_format_violation` across all 4 generations.
* **`code_math` (16 tasks)**: Diagnosed dominant cause as `premature_stop` across all 4 generations.
* **`api_orchestration` (12 tasks)**: Diagnosed dominant cause as `premature_stop` across all 4 generations.

### Mechanism

The diagnoser implements ten ordered structural rules (`DEFAULT_RULES`). It receives only the frozen `AgentSpec` and the `RunRecord` containing the recorded `Trajectory` and `EvaluatorVerdict`:

1. **In `extraction`**: Tasks require structured JSON output matching a target schema. The model frequently produced correct field extractions embedded inside conversational explanations or minor syntax variations. The evaluator scored partial credit based on extracted fields (mean score 0.9136, 0.9303, well above the 0.5 threshold) but failed the binary validation (`passed=False`). The diagnoser fired `_rule_partial_credit` (`score >= 0.5` on a failed run with a final answer present), correctly classifying the failures as `output_format_violation`.
2. **In `code_math`**: Pure arithmetic and code execution tasks. When running without tools or code-execution sandboxes, the model answered calculation prompts directly in a single step (0 tool calls out of an available budget of 8 steps). The diagnoser fired `_rule_stopped_early` (`step_count <= max_steps // 2`), classifying the failures as `premature_stop`.
3. **In `api_orchestration`**: Tasks require multi-hop API interactions. Because tools were not handed to the model by the test harness (see Section 2), the agent concluded after turn 1 without executing tool calls. The diagnoser fired `_rule_stopped_early`, identifying `premature_stop`.

All four registered domains (`code_math`, `api_orchestration`, `extraction`, `mcp_everything`) execute through the exact same five-stage loop in [`agent_engineer/loop.py`](agent_engineer/loop.py) with zero domain-specific branching.

---

## 2. A Silent Harness Bug the Cost Numbers Exposed

In live evaluations of `api_orchestration` ([`artifacts/api_orchestration_gpt5_nano_lineage.md`](artifacts/api_orchestration_gpt5_nano_lineage.md)), the agent achieved **exactly 0.0000 binary accuracy** across all 36 evaluation runs (12 tasks &times; 3 replicates).

The cause was not model inability or excessive task difficulty. It was a silent harness bug that token cost accounting exposed.

### The Cost Discrepancy

Per [`artifacts/cross_domain_live_comparison.md`](artifacts/cross_domain_live_comparison.md), token consumption across domains revealed an impossibility:

| Domain | Task Topology | Expected Steps | Mean Cost / Run |
| :--- | :--- | :---: | :---: |
| **`api_orchestration`** | Multi-hop tool chains (customer &rarr; order &rarr; tracking &rarr; scan) | 2 to 6 hops | **234.7 tokens** |
| **`code_math`** | Single-shot computation | 1 turn | **352.8 tokens** |
| **`extraction`** | Single-shot document parsing | 1 turn | **613.0 tokens** |

The multi-hop orchestration domain had the **lowest token cost of all three domains**, despite being the only domain designed to execute multi-step tool calls.

### The Arithmetic Proof

The run log recorded:
* Total tasks: 12
* Total real model API calls made: **96**
* Number of suite passes: 3 baseline replicates + 1 generation-0 run + 4 mutation evaluations = 8 passes
* Ratio: $96 / 12 = 8.000$ calls per task

The model was called exactly once per task per pass. It never iterated through a multi-turn step loop, never made an intermediate tool call, and never received tool results.

### Root Cause in Code

Investigation of the live-gate driver revealed two silent disconnections:
1. `experiments/run_real_model_gate.py` hardcoded `_NoTools()` for every domain. `spec.tools` was set to `()`, instructing the model to answer directly.
2. `experiments/gpt5_nano_backend.py`'s `OpenAICompatibleBackend.next_action` explicitly discarded tool definitions (`del tools, history`) and sent payloads without a `tools` parameter.

Although tasks in `agent_engineer/domains/api_orchestration/tasks.py` defined `metadata['call_tool']` and `metadata['tool_schemas']`, nothing in the execution driver consumed them. The benchmark was generating plausible numbers (a baseline `mean_score` of 0.0875 from accidental token overlap in direct responses) while measuring zero tool execution.

### Verification and Fix

* **Domain validity**: Hand-built trajectories driving `task.metadata['call_tool']` directly score **1.0000 (12/12 pass)** against the simulated API and evaluator. The domain and evaluator were correctly implemented.
* **Fix**: Landed in commit [`0c703e7`](https://github.com/RedtRocks/aohack/commit/0c703e7812714f25a468ba582955cf476c58d8e1) (PR #13). Added `_MetadataToolRuntime` to dynamically wire tool schemas and dispatch callables from task metadata, and rewrote `OpenAICompatibleBackend` to pass tool schemas, parse tool calls, and maintain message history across steps.

---

## 3. Goodhart's Law Caught in Our Own System

Across the entire live optimization campaign across all three domains (12 candidate generations evaluated), exactly one mutation was accepted: Generation 4 of `extraction` (`memory_reconfigured` to vector retrieval).

The acceptance exposed an objective divergence (Goodhart's Law):

| Metric | Baseline Value | Generation 4 Value | Observed Delta | Noise Floor Threshold | Decision |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Optimization Metric (`mean_score`)** | 0.9136 | 0.9398 | **+0.0262** | 0.015913 | **ACCEPTED** |
| **Headline Metric (`accuracy`)** | 0.8571 | 0.8095 | **-0.0476** | 0.058321 | *(Cannibalized)* |
| **Reliability (variance)** | 0.0317 | 0.0476 | +0.0159 | &mdash; | *(Degraded)* |

### Mechanism and Noise Analysis

1. **The continuous metric improved**: `mean_score` evaluates partial credit (the average fraction of fields extracted correctly across documents). Generation 4 extracted more secondary fields correctly, raising `mean_score` by +0.0262. Because this exceeded the measured noise floor of **0.015913** (measured from baseline replicates `[0.9136, 0.9517, 0.9255]`), the loop's selection policy accepted the mutation.
2. **The binary metric degraded**: `accuracy` requires 100% of fields to match exactly. While Generation 4 extracted more secondary fields on average, it introduced minor formatting errors on 2 previously passing documents, dropping binary accuracy from 0.8571 (12/14 passed) to 0.8095 (11.33/14 passed).
3. **The drop was inside the noise floor**: Binary accuracy across the 3 baseline replicates was `[0.7857 (11/14), 0.9286 (13/14), 0.8571 (12/14)]`, giving an empirical standard deviation of **0.058321**. The accuracy drop of **-0.0476** was smaller than the binary noise floor.

Partial credit improved above its noise floor, while binary completion drifted within sampling noise. Optimizing a composite continuous objective cannibalized strict binary accuracy. This demonstrates why evaluation systems must report multi-metric pairs (accuracy, partial credit, cost, reliability) rather than optimizing a single scalar.

---

## 4. The Diagnosis-to-Mutation Link Was Broken, and We Found It

In the initial implementation of `agent_engineer/stages/mutate.py`, candidate mutations did not always match the diagnosed failure cause.

### The Broken Link

The mutator defined per-cause ladders in `LADDERS`, but had an unconstrained fallback ladder (`_FALLBACK_LADDER`):

```python
_FALLBACK_LADDER = (
    _move_prompt_guidance,
    _move_escalate_strategy,
    _move_raise_step_budget,
    _move_retain_full_context,
    _move_add_episodic_store,
    _move_add_retrieval,
    _move_reorder_tools,
    _move_require_final_answer,
)
```

When a cause's immediate moves were exhausted, the mutator proposed step-budget doublings or vector retrieval for failures diagnosed as `output_format_violation` or `tool_misuse`.

This produced mutations with rationales asserting causes that had never been diagnosed:
* In `extraction` Generation 3 ([`artifacts/extraction_gpt5_nano_lineage.md`](artifacts/extraction_gpt5_nano_lineage.md)), the failure was `output_format_violation`, yet the ladder proposed doubling `max_steps` with the rationale: *"Runs are being cut off at 8 steps before finishing, so the budget is binding..."*
* In `extraction` Generation 4, the ladder proposed vector retrieval with the rationale: *"Full retention did not stop the context loss, so the problem is finding the relevant earlier fact..."*

The loop was proposing unrelated edits under the guise of targeted optimization, behaving like random search with fabricated rationales.

### The Fix

Landed in commit [`b6e8444`](https://github.com/RedtRocks/aohack/commit/b6e844480b3a71f580fdee4d603f50f569d6dc75):
1. Every move on every ladder must carry a rationale that is strictly true of the motivating cause.
2. `_FALLBACK_LADDER` was reduced to two genuinely cause-agnostic moves: prompt guidance and loop-shape strategy escalation.
3. Mutations that have already been evaluated are deduplicated via `(proposal.kind.value, proposal.target_path, proposal.after)`, enabling real strategy branching.
4. When a ladder is exhausted, the mutator returns `None` and the lineage cleanly terminates, rather than proposing irrelevant mutations.

---

## 5. An Honest Negative on Memory, with a Mechanism

The track evaluates: *"Can you show the outputs of the agent getting better over time through its own self-reflection and memory growing?"*

We implemented an explicit `EpisodicMemoryStore` ([`agent_engineer/memory.py`](agent_engineer/memory.py)) that records typed reflections (prompt, verdict, diagnosed failure cause, evidence, and tool sequences) and retrieves relevant past experiences into the prompt context at runtime.

### Empirical Results on Registered Domain (`code_math`)

Measured via `python -m agent_engineer run code_math --memory episodic_store --max-generations 3` ([`artifacts/episodic_memory_growth_demo.md`](artifacts/episodic_memory_growth_demo.md)):

| Generation | Memory Entry Count | Accuracy (with Cost) | Cost / Solved Task | Verdict |
| :--- | :---: | :---: | :---: | :---: |
| **Gen 0 (root)** | 0 entries | 0.1875 (41.9 tokens) | **223.3 tokens** | baseline |
| **Gen 1** | 18 entries | 0.3125 (67.1 tokens) | **214.8 tokens** | ACCEPTED |
| **Gen 2** | 24 entries | 0.6875 (70.9 tokens) | **103.1 tokens** | ACCEPTED |
| **Gen 3** | 29 entries | 0.7500 (71.5 tokens) | **95.3 tokens** | ACCEPTED |

### The Mechanism: Why Memory Does Not Help Single-Turn Tasks

1. **Memory grew**: Reflection entries accumulated from 0 to 29 across iterations.
2. **Token overhead increased**: Retrieving past reflections added **+29.6 tokens/task** (+70.6% prompt overhead, increasing prompt size from 41.9 to 71.5 tokens).
3. **Zero tool calls saved**: On single-turn tasks like `code_math`, tasks finish in a single completion. There are no intermediate exploratory tool calls to eliminate.
4. **Accuracy gain was not driven by memory**: The accuracy gain (0.1875 &rarr; 0.7500) was driven by prompt and strategy mutations accepted by the loop, not by memory retrieval.
5. **Cost per solved task fell only due to pass rate**: Cost per solved task decreased ($223.3 \rightarrow 95.3$ tokens) strictly because more tasks passed, amortizing the prompt cost, not because memory made individual runs cheaper.

### The Architectural Prediction

Episodic memory provides an autonomous cost benefit *only* in multi-hop environments where past reflections prune redundant exploratory tool calls (such as caching discovered endpoints or authentication sequences). On single-turn tasks, memory is a pure token tax.

Empirical validation of this prediction on `api_orchestration` was initially blocked by the tool wiring bug (Section 2) and now awaits live model evaluation with API credentials following the PR #13 fix.

---

## 6. Measurement Discipline Throughout

The system maintains strict measurement discipline across all evaluation paths:

1. **Accuracy denominator includes refusals**: Accuracy is computed over all tasks in the suite. Refusals and errors count as failures, never omitted from the denominator.
2. **Undefined metrics are never zeroed**: When a metric has an empty denominator or insufficient data, it returns `None` and renders as `--`. It is never coerced to `0.0000` or `0.0%`.
3. **Reliability requires 3+ runs**: Reliability is defined strictly as sample variance across repeated runs on the same task. It requires `repeats >= 3` (`MIN_RUNS_FOR_RELIABILITY = 3`). Any attempt to compute reliability with fewer runs returns undefined.
4. **Empirical noise floors**: Before proposing mutations, `run_loop` measures baseline reliability across 3 full suite replicates to compute the empirical standard deviation. Any mutation whose delta falls inside this noise floor is reverted.
5. **Mandatory cost pairing**: In [`agent_engineer/evaluation/report.py`](agent_engineer/evaluation/report.py), reporting functions enforce that accuracy is always paired with token cost (`cost (mean tokens/run)`).
6. **Deterministic grading without LLM judges**:
   * `code_math`: Evaluated by executing submitted code against unit test assertions in an isolated subprocess, with exact numeric tolerance checks.
   * `extraction`: Evaluated by normalized field-by-field string, float, integer, and JSON structure comparison against frozen ground truth.
   * `api_orchestration`: Evaluated by deterministic state verification against the simulated customer support API.
   * `mcp_everything`: Evaluated by protocol compliance and deterministic JSON schema validation against the reference MCP server.

Because there is zero LLM-as-judge scoring anywhere in the codebase, evaluation is 100% deterministic and reproducible with zero judge variance. This is a strictly stronger claim than reporting an inter-rater kappa score.
