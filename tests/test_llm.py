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
    body = json.loads(request.content)
    assert body["max_tokens"] == 8000
    assert body["reasoning"] == {"effort": "low", "exclude": True}


def test_rate_limits_are_retried_honouring_retry_after() -> None:
    sleeps: list[float] = []
    llm = client(
        httpx.Response(429, headers={"retry-after": "7"}),
        httpx.Response(502),
        httpx.Response(200, json=OK),
        sleeps=sleeps,
    )
    assert llm.complete([]).text == '{"a": 1}'
    assert sleeps == [7.0, 2.0]


def test_rate_limits_without_retry_after_back_off_for_longer() -> None:
    sleeps: list[float] = []
    llm = client(*[httpx.Response(429)] * 3, httpx.Response(200, json=OK), sleeps=sleeps)
    assert llm.complete([]).text == '{"a": 1}'
    assert sleeps == [5.0, 15.0, 45.0]
    busy: list[float] = []  # "high demand" 503s wait like rate limits, not like outages
    llm = client(*[httpx.Response(503)] * 2, httpx.Response(200, json=OK), sleeps=busy)
    assert llm.complete([]).text == '{"a": 1}' and busy == [5.0, 15.0]


def test_fallback_models_are_sent_and_upstream_detail_is_reported() -> None:
    seen: list[httpx.Request] = []
    limited = {
        "error": {
            "message": "Provider returned error",
            "metadata": {"provider_name": "Google AI Studio", "raw": "rate-limited upstream"},
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(400, json=limited)

    llm = OpenRouterLLM(
        SecretStr("k"),
        "a:free",
        transport=httpx.MockTransport(handler),
        fallbacks=["b:free", "", "a:free"],
    )
    with pytest.raises(LLMError, match=r"\[Google AI Studio: rate-limited upstream\]"):
        llm.complete([])
    assert json.loads(seen[0].content)["models"] == ["a:free", "b:free"]


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


def test_empty_messages_are_retried_then_explained() -> None:
    empty = {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}
    assert client(httpx.Response(200, json=empty), httpx.Response(200, json=OK)).complete([]).text
    reasoning = {
        "choices": [
            {"message": {"content": None, "reasoning": "thinking..."}, "finish_reason": "length"}
        ]
    }
    llm = client(*[httpx.Response(200, json=reasoning)] * 4)
    with pytest.raises(LLMError) as caught:
        llm.complete([])
    message = str(caught.value)
    assert "finish_reason=length, it returned reasoning only" in message
    assert "raise LLM_MAX_TOKENS" in message


def test_reasoning_can_be_left_out() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=OK)

    llm = OpenRouterLLM(
        SecretStr("k"), "m", transport=httpx.MockTransport(handler), reasoning_effort=None
    )
    llm.complete([])
    assert "reasoning" not in json.loads(seen[0].content)


def test_openai_compatible_providers_get_only_standard_fields() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=OK)

    llm = OpenRouterLLM(
        SecretStr("g-key"),
        "gemini-2.5-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        transport=httpx.MockTransport(handler),
        fallbacks=["other"],
        provider="openai",
    )
    llm.complete([])
    body = json.loads(seen[0].content)
    assert body["reasoning_effort"] == "low"
    assert "reasoning" not in body and "models" not in body
    assert "x-title" not in seen[0].headers and seen[0].headers["authorization"] == "Bearer g-key"
    assert str(seen[0].url).endswith("/v1beta/openai/chat/completions")


def test_googles_list_wrapped_errors_are_reported() -> None:
    wrapped = [{"error": {"code": 400, "message": "API key not valid.", "status": "INVALID"}}]
    with pytest.raises(LLMError, match=r"400: API key not valid\."):
        client(httpx.Response(400, json=wrapped)).complete([])


def _raising(exc: Exception) -> OpenRouterLLM:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return OpenRouterLLM(
        SecretStr("k"), "m", transport=httpx.MockTransport(handler), sleep=lambda _: None
    )


def test_errors_say_whether_trying_again_later_could_help() -> None:
    """copilot.evals record retries only transient errors: timeouts, 429 and 5xx."""
    cases = [
        (client(*[httpx.Response(503, json={"error": {"message": "busy"}})] * 4), 503, True),
        (client(*[httpx.Response(429, json={"error": {"message": "slow"}})] * 4), 429, True),
        (client(httpx.Response(400, json={"error": {"message": "bad schema"}})), 400, False),
        (_raising(httpx.ReadTimeout("slow")), None, True),
        (_raising(httpx.ConnectError("no route")), None, False),
    ]
    for llm, status, transient in cases:
        with pytest.raises(LLMError) as caught:
            llm.complete([{"role": "user", "content": "hi"}])
        assert (caught.value.status, caught.value.transient) == (status, transient), caught.value


def _google_429(retry: str) -> httpx.Response:
    message = (
        "You exceeded your current quota, please check your plan and billing details.\n"
        "* Quota exceeded for metric: generativelanguage.googleapis.com/"
        "generate_content_free_tier_requests, limit: 20, model: gemini-flash\n"
        f"Please retry in {retry}."
    )
    return httpx.Response(429, json=[{"error": {"code": 429, "message": message}}])


def test_an_exhausted_daily_quota_is_not_retried() -> None:
    """Retrying cannot help until the provider resets, and every retry spends a request."""
    for response in (
        _google_429("5h48m4.3s"),
        httpx.Response(
            429, json={"error": {"message": "Rate limit exceeded: free-models-per-day"}}
        ),
    ):
        sleeps: list[float] = []
        llm = client(response, sleeps=sleeps)
        with pytest.raises(LLMError) as caught:
            llm.complete([])
        assert (caught.value.quota, caught.value.transient) == (True, False)
        assert len(llm.seen) == 1 and sleeps == []  # type: ignore[attr-defined]


def test_a_per_minute_quota_is_a_plain_rate_limit() -> None:
    sleeps: list[float] = []
    llm = client(_google_429("41.4s"), httpx.Response(200, json=OK), sleeps=sleeps)
    assert llm.complete([]).text == '{"a": 1}'
    assert sleeps == [5.0]
    with pytest.raises(LLMError) as caught:
        client(*[_google_429("1m3s")] * 4).complete([])
    assert (caught.value.quota, caught.value.transient) == (False, True)


def test_with_retries_off_a_failure_costs_exactly_one_request() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(503, json={"error": {"message": "busy"}})

    llm = OpenRouterLLM(SecretStr("k"), "m", transport=httpx.MockTransport(handler), max_retries=0)
    with pytest.raises(LLMError) as caught:
        llm.complete([])
    assert len(seen) == 1 and caught.value.transient
