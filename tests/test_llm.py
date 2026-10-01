"""The OpenRouter client: retries what can succeed, fails fast on what cannot."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import SecretStr

from copilot.llm import LLMError, OpenRouterLLM, ScriptedLLM

OK = {
    "model": "some/model:free",
    "choices": [{"message": {"content": '{"a": 1}'}}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 3},
}


def client(*responses: httpx.Response, sleeps: list[float] | None = None) -> OpenRouterLLM:
    queue = list(responses)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return queue.pop(0)

    llm = OpenRouterLLM(
        SecretStr("sk-test"),
        "some/model:free",
        transport=httpx.MockTransport(handler),
        sleep=(sleeps.append if sleeps is not None else lambda _: None),
    )
    llm.seen = seen  # type: ignore[attr-defined]
    return llm


def test_a_completion_carries_text_model_and_usage() -> None:
    llm = client(httpx.Response(200, json=OK))
    out = llm.complete([{"role": "user", "content": "hi"}])
    assert (out.text, out.model, out.prompt_tokens, out.completion_tokens) == (
        '{"a": 1}',
        "some/model:free",
        12,
        3,
    )
    request = llm.seen[0]  # type: ignore[attr-defined]
    assert request.headers["authorization"] == "Bearer sk-test"
    assert json.loads(request.content)["max_tokens"] == 3000


def test_rate_limits_are_retried_honouring_retry_after() -> None:
    sleeps: list[float] = []
    llm = client(
        httpx.Response(429, headers={"retry-after": "7"}),
        httpx.Response(503),
        httpx.Response(200, json=OK),
        sleeps=sleeps,
    )
    assert llm.complete([]).text == '{"a": 1}'
    assert sleeps == [7.0, 2.0]


def test_upstream_errors_inside_a_200_are_retried_then_reported() -> None:
    error = {"error": {"code": 502, "message": "Provider returned error"}}
    llm = client(*[httpx.Response(200, json=error)] * 4)
    with pytest.raises(LLMError, match="502: Provider returned error"):
        llm.complete([])


def test_client_errors_are_not_retried_and_never_show_the_key() -> None:
    llm = client(httpx.Response(401, json={"error": {"message": "No auth credentials"}}))
    with pytest.raises(LLMError, match="401: No auth credentials") as caught:
        llm.complete([])
    assert "sk-test" not in str(caught.value) and "sk-test" not in repr(llm.__dict__)


def test_bad_bodies_are_errors() -> None:
    with pytest.raises(LLMError, match="not JSON"):
        client(httpx.Response(200, text="<html>")).complete([])
    with pytest.raises(LLMError, match="unexpected response shape"):
        client(httpx.Response(200, json={"choices": []})).complete([])
    empty = {"choices": [{"message": {"content": "  "}}]}
    with pytest.raises(LLMError, match="empty message"):
        client(httpx.Response(200, json=empty)).complete([])
    with pytest.raises(LLMError, match="400: Bad Request"):
        client(httpx.Response(400, text="nope")).complete([])


def test_unreachable_provider_is_reported_after_retries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    llm = OpenRouterLLM(
        SecretStr("k"), "m", transport=httpx.MockTransport(handler), sleep=lambda _: None
    )
    with pytest.raises(LLMError, match="could not reach the provider: ConnectError"):
        llm.complete([])


def test_a_missing_key_fails_before_any_request() -> None:
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        OpenRouterLLM(SecretStr(""), "m")


def test_scripted_llm_replays_and_records() -> None:
    llm = ScriptedLLM(["one", "two"])
    assert llm.complete([{"role": "user", "content": "a"}]).text == "one"
    assert llm.complete([]).text == "two"
    assert llm.calls[0] == [{"role": "user", "content": "a"}]
    with pytest.raises(LLMError, match="no more responses"):
        llm.complete([])
