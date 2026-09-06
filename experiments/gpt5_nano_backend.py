"""A thin :class:`~agent_engineer.ports.ModelBackend` over a real model: GPT-5 nano,
via the OpenAI-compatible chat-completions API.

This is a standalone experiment script, not part of the installable
``agent_engineer`` package: it exists to answer one question -- does the
engine's loop actually improve a real agent, not just wire together correctly
against a scripted stand-in -- and it is intentionally kept outside the engine
so nothing here can be mistaken for engine code.

The backend is deliberately thin:

* One call per task, always. It never branches on ``spec.strategy``,
  ``spec.memory``, or history -- it reads the spec's ``system_prompt`` and the
  task's ``prompt`` and returns whatever the model said as the final answer.
  A mutation that does not touch ``system_prompt`` text (a strategy change, a
  memory reconfiguration, a stopping-budget raise) is therefore a genuine
  no-op against this backend, and any measured movement from one of those is
  sampling noise, not a real effect -- which is exactly what a real,
  unmodified model should show.
* No domain vocabulary anywhere in this file. The system prompt comes from
  the engine's own :class:`~agent_engineer.stages.synthesize.TemplateSynthesizer`
  and the mutation ladder's own guidance text; this module only transports
  those strings to the API and the API's reply back.
* ``reasoning_effort: "minimal"`` is requested explicitly. GPT-5 nano defaults
  to spending hidden reasoning tokens even on trivial replies (confirmed by a
  smoke call: 74 completion tokens, 64 of them reasoning, for the one-word
  reply "OK"); minimal effort keeps each of the ~150 calls this experiment
  makes fast and cheap without touching what the model is asked to do.
"""

from __future__ import annotations

import os
import time

import requests

from agent_engineer.ports import AgentAction

API_URL = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-5-nano"
TIMEOUT_SECONDS = 60
MAX_RETRIES = 2


class ApiKeyMissing(RuntimeError):
    """Raised when OPENAI_API_KEY is not set. Never simulated around."""


def _api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ApiKeyMissing("OPENAI_API_KEY is not set in the environment")
    return key


class GPT5NanoBackend:
    """Calls real GPT-5 nano once per task. Stateless across calls, no memoization.

    Every call is independent -- run the same (spec, task) pair twice and you
    get two independent samples from the model, which is what lets a later
    step measure real run-to-run variance rather than a cached echo.
    """

    def __init__(self, *, reasoning_effort: str = "minimal") -> None:
        self._key = _api_key()
        self._reasoning_effort = reasoning_effort
        self.calls_made = 0

    def next_action(self, spec, task, tools, history) -> AgentAction:
        del tools, history  # this backend never calls a tool and never re-enters
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                return self._call(spec.system_prompt, task.prompt)
            except Exception as error:  # transient network/API failure
                last_error = error
                if attempt < MAX_RETRIES:
                    time.sleep(1.5 * (attempt + 1))
        assert last_error is not None
        raise last_error

    def _call(self, system_prompt: str, user_prompt: str) -> AgentAction:
        self.calls_made += 1
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "reasoning_effort": self._reasoning_effort,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]["message"]
        usage = body.get("usage", {})
        return AgentAction(
            final_answer=choice.get("content") or "",
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )
