"""Failure handling: auth, missing models, rate limits, timeouts, dead servers.

A free-tier router is a different kind of dependency from a local Ollama: it
runs out of quota on purpose, its providers come and go, and its errors arrive
in OpenAI's envelope. Every one of those has to become a sentence a person can
act on — not a traceback, and not a silent empty string.
"""
import pytest
import requests

from core import llm_client as lc
from tests.conftest import FakeResponse


def _err(status, message, typ="invalid_request_error", headers=None, retry_after=None):
    hdrs = dict(headers or {})
    if retry_after is not None:
        hdrs["Retry-After"] = str(retry_after)
    return FakeResponse(status, {"error": {"message": message, "type": typ}},
                        headers=hdrs)


# ── Authentication ───────────────────────────────────────────────────────────

def test_401_names_the_setting_to_fix(config_file, queue):
    queue(_err(401, "Invalid API key", "authentication_error"))
    config_file({"llm_provider": "freellmapi", "llm_api_key": "wrong-key"})

    with pytest.raises(RuntimeError) as exc:
        lc.call_llm([{"role": "user", "content": "hi"}])

    msg = str(exc.value)
    assert "401" in msg
    assert "llm_api_key" in msg
    assert "wrong-key" not in msg          # never echo the credential back


def test_401_on_the_text_path_too(config_file, queue):
    queue(_err(401, "Invalid API key", "authentication_error"))
    config_file({"llm_provider": "freellmapi", "llm_api_key": "wrong"})

    with pytest.raises(RuntimeError, match="401"):
        lc.call_llm_text("hi")


# ── Missing endpoint / model ─────────────────────────────────────────────────

def test_404_points_at_the_models_endpoint(config_file, queue):
    queue(_err(404, "model 'gpt-9' not found", "invalid_request_error"))
    config_file({"llm_provider": "freellmapi", "llm_model": "gpt-9",
                 "llm_api_key": "k"})

    with pytest.raises(RuntimeError) as exc:
        lc.call_llm_text("hi")

    msg = str(exc.value)
    assert "404" in msg
    assert "/v1/models" in msg


# ── Rate limiting ────────────────────────────────────────────────────────────

def test_429_is_retried_and_then_succeeds(config_file, queue, monkeypatch):
    """FreeLLMAPI answers 429 when every free tier in the pool is momentarily
    exhausted, and says how long to wait. One retry after that is the
    difference between a failed task and a slightly late one."""
    slept = []
    monkeypatch.setattr(lc.time, "sleep", slept.append)
    queue(_err(429, "rate limited", "rate_limit_error", retry_after=2),
          FakeResponse(200, {"choices": [{"message": {"content": "late but fine"}}]}))
    config_file({"llm_provider": "freellmapi", "llm_api_key": "k"})

    out = lc.call_llm([{"role": "user", "content": "hi"}])

    assert out["content"] == "late but fine"
    assert slept == [2.0]                  # honoured the server's Retry-After


def test_429_gives_up_with_an_actionable_message(config_file, queue, monkeypatch):
    monkeypatch.setattr(lc.time, "sleep", lambda _s: None)
    queue(_err(429, "rate limited", "rate_limit_error"))
    config_file({"llm_provider": "freellmapi", "llm_api_key": "k"})

    with pytest.raises(RuntimeError) as exc:
        lc.call_llm_text("hi")

    msg = str(exc.value)
    assert "429" in msg
    assert "rate-limited" in msg


def test_429_retry_count_is_bounded(config_file, queue, monkeypatch):
    """A tight retry loop against a rate limiter only extends the cooldown."""
    monkeypatch.setattr(lc.time, "sleep", lambda _s: None)
    rec = queue(*[_err(429, "rate limited", "rate_limit_error") for _ in range(10)])
    config_file({"llm_provider": "freellmapi", "llm_api_key": "k"})

    with pytest.raises(RuntimeError):
        lc.call_llm_text("hi")

    assert len(rec.posts) == lc._RETRY_LIMIT_ATTEMPTS + 1


# ── Bad requests ─────────────────────────────────────────────────────────────

def test_400_explains_a_bad_routing_profile(config_file, queue):
    """`auto:<profile>` is a real FreeLLMAPI feature, and an unknown profile is
    a documented 400 rather than a silent fallback."""
    queue(_err(400, "unknown routing profile 'coding'", "invalid_request_error"))
    config_file({"llm_provider": "freellmapi", "llm_model": "auto:coding",
                 "llm_api_key": "k"})

    with pytest.raises(RuntimeError) as exc:
        lc.call_llm_text("hi")

    assert "routing profile" in str(exc.value)


def test_non_json_body_is_reported_not_raised_raw(config_file, queue):
    queue(FakeResponse(200, payload=None, text="<html>gateway</html>",
                       raise_on_json=True))
    config_file({"llm_provider": "freellmapi", "llm_api_key": "k"})

    with pytest.raises(RuntimeError, match="non-JSON"):
        lc.call_llm([{"role": "user", "content": "hi"}])


def test_error_envelope_message_is_surfaced(config_file, queue):
    """The router's own wording is the most useful thing in the response."""
    queue(FakeResponse(502, {"error": {"message": "every provider failed",
                                       "type": "provider_error"}}))
    config_file({"llm_provider": "freellmapi", "llm_api_key": "k"})

    with pytest.raises(RuntimeError) as exc:
        lc.call_llm_text("hi")

    assert "every provider failed" in str(exc.value)
    assert "provider_error" in str(exc.value)


# ── Transport ────────────────────────────────────────────────────────────────

def test_connection_error_explains_how_to_start_the_router(config_file, monkeypatch):
    def boom(*a, **kw):
        raise requests.exceptions.ConnectionError("refused")
    monkeypatch.setattr(lc.requests, "post", boom)
    config_file({"llm_provider": "freellmapi", "llm_api_key": "k"})

    with pytest.raises(RuntimeError) as exc:
        lc.call_llm([{"role": "user", "content": "hi"}])

    msg = str(exc.value)
    assert "FreeLLMAPI" in msg
    assert "Keys page" in msg


def test_connection_error_for_openai_keeps_its_own_wording(config_file, monkeypatch):
    def boom(*a, **kw):
        raise requests.exceptions.ConnectionError("refused")
    monkeypatch.setattr(lc.requests, "post", boom)
    config_file({"llm_provider": "openai"})

    with pytest.raises(RuntimeError) as exc:
        lc.call_llm([{"role": "user", "content": "hi"}])

    assert "LM Studio" in str(exc.value)


def test_timeout_is_reported(config_file, monkeypatch):
    def boom(*a, **kw):
        raise requests.exceptions.Timeout("timed out")
    monkeypatch.setattr(lc.requests, "post", boom)
    config_file({"llm_provider": "freellmapi", "llm_api_key": "k"})

    with pytest.raises(RuntimeError, match="timed out"):
        lc.call_llm_text("hi", timeout=7)


def test_streaming_connection_error_is_actionable(config_file, monkeypatch):
    def boom(*a, **kw):
        raise requests.exceptions.ConnectionError("refused")
    monkeypatch.setattr(lc.requests, "post", boom)
    config_file({"llm_provider": "freellmapi", "llm_api_key": "k"})

    with pytest.raises(RuntimeError, match="FreeLLMAPI"):
        list(lc.call_llm_stream([{"role": "user", "content": "hi"}]))


def test_streaming_http_error_carries_the_server_message(config_file, queue):
    queue(_err(429, "rate limited", "rate_limit_error"))
    config_file({"llm_provider": "freellmapi", "llm_api_key": "k"})

    with pytest.raises(RuntimeError) as exc:
        list(lc.call_llm_stream([{"role": "user", "content": "hi"}]))

    assert "rate limited" in str(exc.value)


# ── Ollama behaviour is unchanged ────────────────────────────────────────────

def test_ollama_connection_error_offers_to_start_the_server(config_file, monkeypatch):
    calls = {"n": 0}

    def boom(*a, **kw):
        calls["n"] += 1
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(lc.requests, "post", boom)
    monkeypatch.setattr(lc, "ensure_ollama_running", lambda timeout=15: False)
    config_file({"llm_provider": "ollama"})

    with pytest.raises(RuntimeError) as exc:
        lc.call_llm([{"role": "user", "content": "hi"}])

    assert "ollama serve" in str(exc.value)


def test_ollama_http_error_message_unchanged(config_file, queue):
    queue(FakeResponse(500, None, text="boom"))
    config_file({"llm_provider": "ollama"})

    with pytest.raises(RuntimeError, match="Ollama HTTP error: 500"):
        lc.call_llm([{"role": "user", "content": "hi"}])


# ── Health probing ───────────────────────────────────────────────────────────

def test_ensure_running_pings_the_models_endpoint_for_freellmapi(config_file, queue):
    rec = queue(FakeResponse(200, {"data": []}))
    config_file({"llm_provider": "freellmapi", "llm_api_key": "k"})

    assert lc.ensure_ollama_running() is True
    assert rec.gets[0]["url"] == "http://localhost:3001/v1/models"


def test_diagnose_reports_an_unreachable_server(config_file, monkeypatch):
    def boom(*a, **kw):
        raise requests.exceptions.ConnectionError("refused")
    monkeypatch.setattr(lc.requests, "get", boom)
    config_file({"llm_provider": "freellmapi", "llm_api_key": "k"})

    info = lc.diagnose()

    assert info["reachable"] is False
    assert "ConnectionError" in info["error"]
