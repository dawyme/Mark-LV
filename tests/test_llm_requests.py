"""The bytes that go on the wire, per provider.

This is the test that would have caught the original bug: `call_llm_text()`
built an Ollama `/api/chat` payload unconditionally, so the moment
"llm_provider" was set to anything else it posted an Ollama-shaped body to a
server that had never heard of `keep_alive` or `num_predict`.

Each test here asserts the endpoint, the payload keys and the auth header —
against a fake transport, so it is the request format that is under test and
not somebody's network.
"""
import pytest

from core import llm_client as lc
from tests.conftest import FakeResponse


def _chat_reply(content="Hello there.", tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return FakeResponse(200, {"choices": [{"message": msg, "finish_reason": "stop"}]})


# ── call_llm: OpenAI-compatible wire format ──────────────────────────────────

def test_freellmapi_chat_request_shape(config_file, queue):
    rec = queue(_chat_reply())
    config_file({"llm_provider": "freellmapi",
                 "llm_api_key": "freellmapi-key-1"})

    out = lc.call_llm([{"role": "user", "content": "hi"}])

    call = rec.posts[0]
    assert call["url"] == "http://localhost:3001/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer freellmapi-key-1"
    body = call["json"]
    assert body["model"] == "auto"
    assert body["stream"] is False
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["max_tokens"] == 150
    # Ollama-only fields must never reach an OpenAI-compatible server.
    assert "keep_alive" not in body
    assert "options" not in body
    assert out["content"] == "Hello there."
    assert out["tool_calls"] == []


def test_openai_chat_request_shape_is_the_same(config_file, queue):
    rec = queue(_chat_reply())
    config_file({"llm_provider": "openai", "llm_model": "qwen2.5-7b",
                 "llm_api_key": "lm-studio-key"})

    lc.call_llm([{"role": "user", "content": "hi"}])

    call = rec.posts[0]
    assert call["url"] == "http://localhost:1234/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer lm-studio-key"
    assert call["json"]["model"] == "qwen2.5-7b"


def test_openai_server_without_a_key_sends_no_auth_header(config_file, queue):
    """LM Studio on localhost needs no key, and an empty Bearer would be worse."""
    rec = queue(_chat_reply())
    config_file({"llm_provider": "openai"})

    lc.call_llm([{"role": "user", "content": "hi"}])

    assert "Authorization" not in rec.posts[0]["headers"]


def test_tools_are_forwarded_with_tool_choice_auto(config_file, queue):
    rec = queue(_chat_reply())
    config_file({"llm_provider": "freellmapi", "llm_api_key": "k"})
    tools = [{"type": "function",
              "function": {"name": "get_weather", "parameters": {}}}]

    lc.call_llm([{"role": "user", "content": "weather?"}], tools=tools)

    body = rec.posts[0]["json"]
    assert body["tools"] == tools
    assert body["tool_choice"] == "auto"


def test_no_tools_key_when_none_are_given(config_file, queue):
    rec = queue(_chat_reply())
    config_file({"llm_provider": "freellmapi", "llm_api_key": "k"})

    lc.call_llm([{"role": "user", "content": "hi"}])

    assert "tools" not in rec.posts[0]["json"]
    assert "tool_choice" not in rec.posts[0]["json"]


# ── call_llm: Ollama wire format is untouched ────────────────────────────────

def test_ollama_chat_request_shape_unchanged(config_file, queue):
    rec = queue(FakeResponse(200, {"message": {"content": "Ollama here."}}))
    config_file({"llm_provider": "ollama"})

    out = lc.call_llm([{"role": "user", "content": "hi"}])

    call = rec.posts[0]
    assert call["url"] == "http://localhost:11434/api/chat"
    body = call["json"]
    assert body["model"] == "llama3.2"
    assert body["keep_alive"] == -1
    assert body["options"] == {"num_predict": 150, "num_gpu": 99}
    assert body["stream"] is False
    assert "max_tokens" not in body          # an OpenAI field, not Ollama's
    assert out["content"] == "Ollama here."


def test_ollama_tool_calls_pass_through_unchanged(config_file, queue):
    """Ollama already answers with dict arguments; nothing may reshape them."""
    rec = queue(FakeResponse(200, {"message": {
        "content": "",
        "tool_calls": [{"function": {"name": "open_app",
                                     "arguments": {"app": "notepad"}}}],
    }}))
    config_file({"llm_provider": "ollama"})

    out = lc.call_llm([{"role": "user", "content": "open notepad"}])

    assert out["tool_calls"][0]["function"]["arguments"] == {"app": "notepad"}


# ── call_llm_text: the provider-compatibility fix ────────────────────────────

def test_call_llm_text_uses_the_openai_surface_for_freellmapi(config_file, queue):
    """The regression this file exists for."""
    rec = queue(_chat_reply("A plan with three steps."))
    config_file({"llm_provider": "freellmapi", "llm_api_key": "k"})

    text = lc.call_llm_text("Break this into steps", system="You are a planner.")

    call = rec.posts[0]
    assert call["url"] == "http://localhost:3001/v1/chat/completions"
    body = call["json"]
    assert body["messages"][0] == {"role": "system", "content": "You are a planner."}
    assert body["messages"][1]["role"] == "user"
    assert body["max_tokens"] == 600
    assert "keep_alive" not in body
    assert text == "A plan with three steps."


def test_call_llm_text_uses_the_openai_surface_for_openai(config_file, queue):
    rec = queue(_chat_reply("done"))
    config_file({"llm_provider": "openai", "llm_model": "m1"})

    lc.call_llm_text("prompt")

    assert rec.posts[0]["url"].endswith("/v1/chat/completions")


def test_call_llm_text_model_override(config_file, queue):
    rec = queue(_chat_reply("ok"))
    config_file({"llm_provider": "freellmapi", "llm_model": "auto", "llm_api_key": "k"})

    lc.call_llm_text("prompt", model="auto:smart")

    assert rec.posts[0]["json"]["model"] == "auto:smart"


def test_call_llm_text_still_uses_ollama_for_ollama(config_file, queue):
    rec = queue(FakeResponse(200, {"message": {"content": "ollama text"}}))
    config_file({"llm_provider": "ollama"})

    assert lc.call_llm_text("prompt") == "ollama text"
    assert rec.posts[0]["url"].endswith("/api/chat")


# ── Tool-call normalisation ──────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ('{"city": "Karachi"}', {"city": "Karachi"}),
    ('  {"city":"Lagos"}  ', {"city": "Lagos"}),
    ('{"city": "Lagos"}', {"city": "Lagos"}),
    ('prose {"city": "Lagos"} prose', {"city": "Lagos"}),   # salvage the object
    ('{"city": "Lagos",}', '{"city": "Lagos",}'),            # unsalvageable → raw
    ('not json at all', "not json at all"),
    ('', {}),
    (None, {}),
    ({"already": "a dict"}, {"already": "a dict"}),
])
def test_tool_argument_parsing(raw, expected):
    assert lc._parse_tool_arguments(raw) == expected


def test_openai_tool_calls_are_normalised_to_ollama_shape(config_file, queue):
    rec = queue(_chat_reply("", tool_calls=[
        {"id": "call_1", "type": "function",
         "function": {"name": "get_weather", "arguments": '{"city": "Karachi"}'}},
    ]))
    config_file({"llm_provider": "freellmapi", "llm_api_key": "k"})

    out = lc.call_llm([{"role": "user", "content": "weather?"}], tools=[{"x": 1}])

    tc = out["tool_calls"]
    assert len(tc) == 1
    assert tc[0]["id"] == "call_1"
    assert tc[0]["function"]["name"] == "get_weather"
    assert tc[0]["function"]["arguments"] == {"city": "Karachi"}


def test_malformed_tool_arguments_do_not_raise(config_file, queue):
    """A half-streamed argument string must come back as data, not an exception."""
    rec = queue(_chat_reply("", tool_calls=[
        {"id": "call_2",
         "function": {"name": "reminder", "arguments": '{"when": "9am", "what":'}},
    ]))
    config_file({"llm_provider": "freellmapi", "llm_api_key": "k"})

    out = lc.call_llm([{"role": "user", "content": "remind me"}], tools=[{"x": 1}])

    assert out["tool_calls"][0]["function"]["name"] == "reminder"
    assert out["tool_calls"][0]["function"]["arguments"] == '{"when": "9am", "what":'


def test_tool_call_fragment_without_a_name_is_dropped(config_file, queue):
    rec = queue(_chat_reply("hi", tool_calls=[
        {"id": "", "function": {"name": "", "arguments": "{}"}},
    ]))
    config_file({"llm_provider": "freellmapi", "llm_api_key": "k"})

    out = lc.call_llm([{"role": "user", "content": "hi"}], tools=[{"x": 1}])

    assert out["tool_calls"] == []


# ── Model listing / health ───────────────────────────────────────────────────

def test_list_models_uses_the_documented_models_endpoint(config_file, queue):
    rec = queue(FakeResponse(200, {"object": "list", "data": [
        {"id": "auto"}, {"id": "fusion"}, {"id": "gemini-2.5-flash"},
    ]}))
    config_file({"llm_provider": "freellmapi", "llm_api_key": "k"})

    models = lc.list_models()

    assert rec.gets[0]["url"] == "http://localhost:3001/v1/models"
    assert rec.gets[0]["headers"]["Authorization"] == "Bearer k"
    assert models == ["auto", "fusion", "gemini-2.5-flash"]


def test_check_model_available_accepts_the_router_virtual_entries(config_file, queue):
    """`auto`, `auto:*` and `fusion` are router entries, not catalog rows."""
    queue(FakeResponse(200, {"data": []}))
    config_file({"llm_provider": "freellmapi", "llm_api_key": "k"})

    for model in ("auto", "auto:fast", "auto:coding", "fusion"):
        config_file({"llm_provider": "freellmapi", "llm_model": model,
                     "llm_api_key": "k"})
        assert lc.check_model_available() is True, model


def test_check_model_available_flags_a_model_outside_the_catalog(config_file, queue):
    rec = queue(FakeResponse(200, {"data": [{"id": "gemini-2.5-flash"}]}))
    config_file({"llm_provider": "freellmapi", "llm_model": "gpt-9-turbo",
                 "llm_api_key": "k"})

    logged = []
    assert lc.check_model_available(log=logged.append) is False
    assert any("not in the FreeLLMAPI catalog" in line for line in logged)


def test_diagnose_redacts_the_key(config_file, queue):
    queue(FakeResponse(200, {"data": [{"id": "auto"}]}))
    config_file({"llm_provider": "freellmapi", "llm_api_key": "freellmapi-secret"})

    info = lc.diagnose()

    assert info["provider"] == "freellmapi"
    assert info["base_url"] == "http://localhost:3001/v1"
    assert info["model"] == "auto"
    assert info["api_key_set"] is True
    assert info["reachable"] is True
    assert "freellmapi-secret" not in str(info)
