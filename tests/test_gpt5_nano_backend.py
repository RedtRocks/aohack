"""Tests for experiments/gpt5_nano_backend.py: the real-model ModelBackend.

None of these make a network call. They exercise the wiring that matters --
missing-key handling, response parsing into an AgentAction, and the retry
path -- with ``requests.post`` mocked out. The backend is unused by default:
nothing in the main test suite or in ``agent_engineer`` itself imports it, and
it only does anything once a caller sets ``OPENAI_API_KEY`` and constructs it
explicitly (see ``experiments/run_real_model_gate.py`` and the README).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from gpt5_nano_backend import (  # noqa: E402
    ApiKeyMissing,
    FallbackBackend,
    GPT5NanoBackend,
    TensorMuxGLMBackend,
)


class _Task:
    def __init__(self, prompt: str) -> None:
        self.prompt = prompt


class _Spec:
    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt


def _fake_response(*, content: str, prompt_tokens: int = 12, completion_tokens: int = 7) -> Mock:
    response = Mock()
    response.raise_for_status = Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }
    return response


def test_missing_key_raises_without_simulating_one(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ApiKeyMissing):
        GPT5NanoBackend()


def test_next_action_parses_a_real_response_shape(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    backend = GPT5NanoBackend()

    post = Mock(return_value=_fake_response(content="def f(): return 1"))
    monkeypatch.setattr("gpt5_nano_backend.requests.post", post)

    action = backend.next_action(_Spec("system text"), _Task("user text"), (), ())

    assert action.final_answer == "def f(): return 1"
    assert action.prompt_tokens == 12
    assert action.completion_tokens == 7
    assert action.is_final
    assert backend.calls_made == 1

    # the system/user split reaches the API untouched, and reasoning_effort is set
    _, kwargs = post.call_args
    body = kwargs["json"]
    assert body["messages"] == [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "user text"},
    ]
    assert body["reasoning_effort"] == "minimal"


def test_next_action_retries_transient_failures_then_succeeds(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    backend = GPT5NanoBackend()
    monkeypatch.setattr("gpt5_nano_backend.time.sleep", lambda _seconds: None)

    post = Mock(
        side_effect=[
            ConnectionError("transient"),
            _fake_response(content="ok"),
        ]
    )
    monkeypatch.setattr("gpt5_nano_backend.requests.post", post)

    action = backend.next_action(_Spec("s"), _Task("u"), (), ())

    assert action.final_answer == "ok"
    assert post.call_count == 2


def test_next_action_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    backend = GPT5NanoBackend()
    monkeypatch.setattr("gpt5_nano_backend.time.sleep", lambda _seconds: None)

    post = Mock(side_effect=ConnectionError("still down"))
    monkeypatch.setattr("gpt5_nano_backend.requests.post", post)

    with pytest.raises(ConnectionError):
        backend.next_action(_Spec("s"), _Task("u"), (), ())


def test_tensormux_backend_missing_key_raises(monkeypatch):
    monkeypatch.delenv("TENSORMUX_API_KEY", raising=False)
    with pytest.raises(ApiKeyMissing):
        TensorMuxGLMBackend()


def test_tensormux_backend_calls_its_own_base_url_and_model(monkeypatch):
    monkeypatch.setenv("TENSORMUX_API_KEY", "tmx-test-not-a-real-key")
    backend = TensorMuxGLMBackend()

    post = Mock(return_value=_fake_response(content="glm reply"))
    monkeypatch.setattr("gpt5_nano_backend.requests.post", post)

    action = backend.next_action(_Spec("s"), _Task("u"), (), ())

    assert action.final_answer == "glm reply"
    args, kwargs = post.call_args
    assert args[0] == "https://api.tensormux.com/v1/chat/completions"
    assert kwargs["json"]["model"] == "glm-4-7-flash"
    # TensorMux is not asked for reasoning_effort -- that field is GPT-5-nano-specific
    assert "reasoning_effort" not in kwargs["json"]
    # the auth header carries the key but nothing here ever logs or returns it
    assert kwargs["headers"]["Authorization"] == "Bearer tmx-test-not-a-real-key"
    # GLM's hidden reasoning trace is disabled: it otherwise consumes the whole
    # max_tokens budget before any visible content is written (confirmed live:
    # max_tokens=400 gave finish_reason="length" and content=null)
    assert kwargs["json"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert kwargs["json"]["max_tokens"] == 400


def test_fallback_backend_uses_primary_when_it_succeeds(monkeypatch):
    monkeypatch.setenv("TENSORMUX_API_KEY", "tmx-test-not-a-real-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    primary = TensorMuxGLMBackend()
    secondary = GPT5NanoBackend()
    fallback = FallbackBackend(primary, secondary)

    post = Mock(return_value=_fake_response(content="primary answered"))
    monkeypatch.setattr("gpt5_nano_backend.requests.post", post)

    action = fallback.next_action(_Spec("s"), _Task("u"), (), ())

    assert action.final_answer == "primary answered"
    assert fallback.primary_calls == 1
    assert fallback.secondary_calls == 0
    assert post.call_count == 1


def test_fallback_backend_falls_back_when_primary_is_exhausted(monkeypatch):
    monkeypatch.setenv("TENSORMUX_API_KEY", "tmx-test-not-a-real-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    primary = TensorMuxGLMBackend()
    secondary = GPT5NanoBackend()
    fallback = FallbackBackend(primary, secondary)
    monkeypatch.setattr("gpt5_nano_backend.time.sleep", lambda _seconds: None)

    # every primary attempt fails (all of MAX_RETRIES+1 = 3), then the
    # secondary succeeds on its first attempt
    responses = [ConnectionError("tensormux down")] * 3 + [_fake_response(content="fallback answered")]
    post = Mock(side_effect=responses)
    monkeypatch.setattr("gpt5_nano_backend.requests.post", post)

    action = fallback.next_action(_Spec("s"), _Task("u"), (), ())

    assert action.final_answer == "fallback answered"
    assert fallback.primary_calls == 0
    assert fallback.secondary_calls == 1


def test_fallback_backend_raises_when_both_providers_fail(monkeypatch):
    monkeypatch.setenv("TENSORMUX_API_KEY", "tmx-test-not-a-real-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    primary = TensorMuxGLMBackend()
    secondary = GPT5NanoBackend()
    fallback = FallbackBackend(primary, secondary)
    monkeypatch.setattr("gpt5_nano_backend.time.sleep", lambda _seconds: None)

    post = Mock(side_effect=ConnectionError("everything is down"))
    monkeypatch.setattr("gpt5_nano_backend.requests.post", post)

    with pytest.raises(ConnectionError):
        fallback.next_action(_Spec("s"), _Task("u"), (), ())
