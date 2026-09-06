# Diagnosis-to-Mutation Link Audit (Pre-PR 12 Lineages)

This audit analyzes all 12 candidate generations across the three pre-PR 12 live lineages (`code_math`, `api_orchestration`, `extraction`), evaluating whether each proposed mutation plausibly addressed the diagnosed dominant cause.

## Audit Table

| Generation | Domain | Stated Dominant Cause | Mutation Kind Applied | Plausibly Addressed? | Stated Rationale vs. Diagnosed Reality |
|:---:|:---|:---|:---|:---:|:---|
| 1 | `code_math` | `premature_stop` | `system_prompt_rewrite` | **Plausible** | Added behavioral guidance directing the agent to verify before stopping. |
| 2 | `code_math` | `premature_stop` | `strategy_changed` | **Plausible** | Shifted to `plan_then_execute` to force a planning step before completion. |
| 3 | `code_math` | `premature_stop` | `stopping_adjusted` | **Broken** | Rationale asserted *"runs are being cut off at 8 steps"*; agent actually stopped on step 1. Doubling step budget does not fix premature stopping. |
| 4 | `code_math` | `premature_stop` | `memory_reconfigured` | **Broken** | Rationale asserted *"full retention did not stop the context loss"*; context loss was never diagnosed for this calculation domain. |
| 1 | `api_orchestration` | `premature_stop` | `system_prompt_rewrite` | **Plausible** | Added prompt guidance for completing multi-hop requirements before terminating. |
| 2 | `api_orchestration` | `premature_stop` | `strategy_changed` | **Plausible** | Escalated loop shape to `plan_then_execute`. |
| 3 | `api_orchestration` | `premature_stop` | `stopping_adjusted` | **Broken** | Rationale asserted step budget cut-off; the agent stopped on turn 1 due to missing tool execution, never exhausting its budget. |
| 4 | `api_orchestration` | `premature_stop` | `memory_reconfigured` | **Broken** | Rationale asserted context loss retrieval for premature stopping. |
| 1 | `extraction` | `output_format_violation` | `system_prompt_rewrite` | **Plausible** | Added prompt instructions emphasizing strict schema and output shape formatting. |
| 2 | `extraction` | `output_format_violation` | `strategy_changed` | **Broken** | Shifted strategy to `plan_then_execute`; planning deliberation does not resolve schema/JSON syntax formatting errors (PR 12 replaced this with `reflexion`). |
| 3 | `extraction` | `output_format_violation` | `stopping_adjusted` | **Broken** | Doubled step budget (8 -> 16) with budget cut-off rationale for a pure single-turn JSON syntax violation. |
| 4 | `extraction` | `output_format_violation` | `memory_reconfigured` | **Broken** | Reconfigured memory to vector retrieval over 6 items with context loss rationale; document extraction does not suffer from inter-step context loss. |

## Empirical Evidence of the Broken Link

1. **Unconstrained Fallback Ladder (`_FALLBACK_LADDER`)**:
   Before PR 12, when a failure cause exhausted its immediate ladder moves (typically after Gen 1 or 2), the mutator fell back to a hardcoded list containing `_move_raise_step_budget` and `_move_retain_full_context` / `_move_add_retrieval`.
2. **Fabricated Rationales**:
   Every domain's Generation 3 asserted *"Runs are being cut off at 8 steps before finishing..."*, regardless of whether runs stopped on step 1 (`premature_stop`) or failed JSON parsing (`output_format_violation`).
   Every domain's Generation 4 asserted *"Full retention did not stop the context loss..."*, regardless of whether the domain had tools, multi-turn history, or context degradation.
3. **Audit Verdict**:
   Out of 12 candidate generations, **7 out of 12 (58.3%)** mutations carried rationales completely detached from the diagnosed failure cause, acting as unguided random search rather than targeted closed-loop optimization.
