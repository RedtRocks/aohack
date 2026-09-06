# api_orchestration: Memory Control vs. Treatment Experiment

Live experiment on `api_orchestration` using real model execution (`glm-4-7-flash` via TensorMux primary, `gpt-5-nano` fallback) and real tool calling via `ApiOrchestrationToolRuntime`.

| arm | accuracy (with cost) | tool calls per solved task | noise floor |
| :--- | :---: | :---: | :---: |
| **control (memory OFF)** | 0.7222 (6104.8 tok/run) | 3.1154 (81 calls / 26 solved) | std 0.071827 (var 0.005159) |
| **treatment (memory ON)** | not measured | not measured | not measured |

* **Prediction under test**: Whether tool calls per solved task fell with episodic memory is **not measured** (treatment arm stopped before completion).
* **api_orchestration new baseline**: With real tool execution enabled via PR #13, `api_orchestration` achieved a real baseline of **0.7222 accuracy (26/36 tasks solved, 6104.8 tokens/run)**, replacing the previous artificial 0.0000 score caused by unwired tools.
