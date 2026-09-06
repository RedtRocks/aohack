"""Thin :class:`~agent_engineer.ports.ModelBackend` implementations over real
models, via OpenAI-compatible chat-completions APIs, plus a fallback wrapper
that tries a primary provider first.

This is a standalone experiment module, not part of the installable
``agent_engineer`` package: it exists to answer one question -- does the
engine's loop actually improve a real agent, not just wire together correctly
against a scripted stand-in -- and it is intentionally kept outside the engine
so nothing here can be mistaken for engine code.

Every backend here is deliberately thin:

* One call per task, always. It never branches on ``spec.strategy``,
  ``spec.memory``, or history -- it reads the spec's ``system_prompt`` and the
  task's ``prompt`` and returns whatever the model said as the final answer.
  A mutation that does not touch ``system_prompt`` text (a strategy change, a
  memory reconfiguration, a stopping-budget raise) is therefore a genuine
  no-op against these backends, and any measured movement from one of those
  is sampling noise, not a real effect -- which is exactly what a real,
  unmodified model should show.
* No domain vocabulary anywhere in this file. The system prompt comes from
  the engine's own :class:`~agent_engineer.stages.synthesize.TemplateSynthesizer`
  and the mutation ladder's own guidance text; this module only transports
  those strings to the API and the API's reply back.
* API keys are read from environment variables at call time, never
  hardcoded, never logged, and never included in any exception message or
  request log -- only the auth header carries one, and nothing here prints
  headers.

Two providers are wired:

* :class:`TensorMuxGLMBackend` -- GLM 4.7 Flash via TensorMux, the primary.
  Reads ``TENSORMUX_API_KEY``.
* :class:`GPT5NanoBackend` -- GPT-5 nano via the standard OpenAI API, the
  fallback. Reads ``OPENAI_API_KEY``. ``reasoning_effort: "minimal"`` is
  requested explicitly: GPT-5 nano defaults to spending hidden reasoning
  tokens even on trivial replies (confirmed by a smoke call: 74 completion
  tokens, 64 of them reasoning, for the one-word reply "OK"); minimal effort
  keeps each call fast and cheap without touching what the model is asked.

:class:`FallbackBackend` tries the primary for every call and only falls back
to the secondary once the primary has exhausted its own retries -- for a
provider outage or a rate limit, not for an occasional blip.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

from agent_engineer.ports import AgentAction

TIMEOUT_SECONDS = 60
MAX_RETRIES = 1
"""One retry maximum -- a call either works on the second try or the whole run
stops and reports, rather than looping on a provider that is down."""

DEFAULT_MAX_TOKENS = 400
"""code_math answers are short (a number, or one short function). Capped
tightly so a model that rambles cannot silently multiply the bill."""


class ApiKeyMissing(RuntimeError):
    """Raised when a required API key env var is not set. Never simulated around."""


class OpenAICompatibleBackend:
    """Calls a real model once per task over an OpenAI-compatible chat-completions
    API. Stateless across calls, no memoization: every call is an independent
    sample from the model, which is what lets a later step measure real
    run-to-run variance rather than a cached echo.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str,
        reasoning_effort: str | None = None,
        disable_thinking: bool = False,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        key = os.environ.get(api_key_env)
        if not key:
            raise ApiKeyMissing(f"{api_key_env} is not set in the environment")
        self._key = key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._disable_thinking = disable_thinking
        self._max_tokens = max_tokens
        self.calls_made = 0
        self.total_tokens = 0

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
        payload = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort
        if self._disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        response = requests.post(
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]["message"]
        usage = body.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        self.total_tokens += prompt_tokens + completion_tokens
        return AgentAction(
            final_answer=choice.get("content") or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class GPT5NanoBackend(OpenAICompatibleBackend):
    """GPT-5 nano over the standard OpenAI API. Reads ``OPENAI_API_KEY``."""

    def __init__(self, *, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        super().__init__(
            base_url="https://api.openai.com/v1",
            model="gpt-5-nano",
            api_key_env="OPENAI_API_KEY",
            reasoning_effort="minimal",
            max_tokens=max_tokens,
        )


class TensorMuxGLMBackend(OpenAICompatibleBackend):
    """GLM 4.7 Flash via TensorMux's OpenAI-compatible endpoint.
    Reads ``TENSORMUX_API_KEY``."""

    def __init__(self, *, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        super().__init__(
            base_url="https://api.tensormux.com/v1",
            model="glm-4-7-flash",
            api_key_env="TENSORMUX_API_KEY",
            # GLM 4.7 Flash defaults to a hidden reasoning trace before any visible
            # content -- confirmed by a smoke call: max_tokens=400 was entirely
            # consumed by reasoning, finish_reason "length", content null. Disabling
            # it is a ~7x cost cut AND the difference between getting an answer at
            # all and getting truncated silence; it is a universal generation
            # parameter, not a per-task or per-domain tweak.
            disable_thinking=True,
            max_tokens=max_tokens,
        )


class FallbackBackend:
    """Tries ``primary`` for every call; only tries ``secondary`` once ``primary``
    has exhausted its own retries (an outage or a rate limit), not for an
    occasional blip. Reports which provider actually answered each call."""

    def __init__(self, primary, secondary) -> None:
        self._primary = primary
        self._secondary = secondary
        self.primary_calls = 0
        self.secondary_calls = 0

    @property
    def calls_made(self) -> int:
        return getattr(self._primary, "calls_made", 0) + getattr(self._secondary, "calls_made", 0)

    @property
    def total_tokens(self) -> int:
        return getattr(self._primary, "total_tokens", 0) + getattr(self._secondary, "total_tokens", 0)

    def next_action(self, spec, task, tools, history) -> AgentAction:
        try:
            action = self._primary.next_action(spec, task, tools, history)
            self.primary_calls += 1
            return action
        except Exception as primary_error:
            try:
                action = self._secondary.next_action(spec, task, tools, history)
                self.secondary_calls += 1
                return action
            except Exception as secondary_error:
                raise secondary_error from primary_error


class CachingBackend:
    """Wraps another backend with a crash-safe, on-disk result cache, so a rerun
    after a crash (or a deliberate resumption) never re-pays for a call it
    already made and recorded.

    The cache key is ``(spec_id, task_id, call_index)`` where ``call_index``
    counts how many times *this process* has already asked for that exact
    (spec, task) pair. That makes it safe for the two shapes this experiment
    needs: a generation's single call per task (call_index is always 0, so a
    rerun after the same generation replays the cached answer instead of
    re-spending) and a noise-floor probe's several independent replicate calls
    on the very same spec (call_index 0, 1, 2, ... are each their own cache
    slot, so replicate 2 is never accidentally served replicate 0's answer).
    Never used to collapse genuinely independent samples into one.
    """

    def __init__(self, backend, cache_path: Path) -> None:
        self._backend = backend
        self._cache_path = cache_path
        self._counts: dict[str, int] = {}
        self._cache: dict[str, dict] = {}
        if cache_path.exists():
            self._cache = json.loads(cache_path.read_text(encoding="utf-8"))

    @property
    def calls_made(self) -> int:
        return getattr(self._backend, "calls_made", 0)

    @property
    def total_tokens(self) -> int:
        return getattr(self._backend, "total_tokens", 0)

    def next_action(self, spec, task, tools, history) -> AgentAction:
        pair_key = f"{spec.spec_id}::{task.task_id}"
        index = self._counts.get(pair_key, 0)
        self._counts[pair_key] = index + 1
        cache_key = f"{pair_key}::{index}"

        cached = self._cache.get(cache_key)
        if cached is not None:
            return AgentAction(
                final_answer=cached["final_answer"],
                prompt_tokens=cached["prompt_tokens"],
                completion_tokens=cached["completion_tokens"],
            )

        action = self._backend.next_action(spec, task, tools, history)
        self._cache[cache_key] = {
            "final_answer": action.final_answer,
            "prompt_tokens": action.prompt_tokens,
            "completion_tokens": action.completion_tokens,
        }
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(self._cache), encoding="utf-8")
        return action
