"""The free rung on the Gemini ladder.

core/gemini.py is the single funnel every one-shot model call in the app goes
through — dev_agent, code_helper, file_processor, flight_finder, desktop,
computer_control, web_search. Putting FreeLLMAPI at the BOTTOM of that ladder
is what lets all of them survive an exhausted Gemini text quota without a
single one of them being edited.

These tests pin the two properties that matter: the rung is last (Gemini always
gets the first refusal) and it is off unless it was configured.
"""
import pytest

from core import gemini
from core import llm_client as lc


class _FakeParts:
    """Stands in for a google.genai Part carrying an image."""

    def __init__(self, data=b"\x89PNG", mime="image/png"):
        class _Blob:
            pass
        self.inline_data = _Blob()
        self.inline_data.data = data
        self.inline_data.mime_type = mime
        self.text = None


class _RaisingClient:
    """Every Gemini rung on the ladder is out of quota."""

    class _Models:
        def generate_content(self, **kwargs):
            raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")

    def __init__(self):
        self.models = _RaisingClient()


@pytest.fixture
def gemini_env(tmp_path, monkeypatch):
    """A scratch api_keys.json for core.gemini, a stubbed Live rung, and a
    stubbed Gemini client that always 429s — so whatever answers did not come
    from Gemini."""
    cfg = tmp_path / "api_keys.json"
    cfg.write_text('{"gemini_api_key": "fake-gemini-key"}', encoding="utf-8")
    monkeypatch.setattr(gemini, "_KEY_FILE", cfg)
    monkeypatch.setattr(gemini, "_cached_key", None)
    monkeypatch.setattr(gemini, "client", lambda **kw: _RaisingClient())
    monkeypatch.setattr(gemini, "_live_call", lambda *a, **kw: None)
    gemini._cooldown.clear()
    return cfg


@pytest.fixture
def llm_env(tmp_path, monkeypatch):
    """Point core.llm_client at the same scratch config, env cleared."""
    path = tmp_path / "api_keys.json"
    monkeypatch.setattr(lc, "CONFIG_PATH", path)
    for var in ("MARK_LLM_PROVIDER", "MARK_LLM_URL", "MARK_LLM_MODEL",
                "MARK_LLM_API_KEY", "FREELLMAPI_API_KEY", "FREELLMAPI_BASE_URL",
                "FREELLMAPI_MODEL", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    return path


def _write(cfg_path, llm_path, data):
    import json
    merged = {"gemini_api_key": "fake-gemini-key"}
    merged.update(data)
    cfg_path.write_text(json.dumps(merged), encoding="utf-8")
    llm_path.write_text(json.dumps(merged), encoding="utf-8")


# ── The rung is opt-in ───────────────────────────────────────────────────────

def test_rung_is_off_when_the_provider_is_ollama(gemini_env, llm_env):
    _write(gemini_env, llm_env, {"llm_provider": "ollama"})
    assert gemini.freellmapi_enabled() is False


def test_rung_is_off_without_a_key(gemini_env, llm_env):
    _write(gemini_env, llm_env, {"llm_provider": "freellmapi"})
    assert gemini.freellmapi_enabled() is False


def test_rung_is_on_when_configured(gemini_env, llm_env):
    _write(gemini_env, llm_env,
           {"llm_provider": "freellmapi", "llm_api_key": "freellmapi-key"})
    assert gemini.freellmapi_enabled() is True


def test_rung_can_be_switched_off_explicitly(gemini_env, llm_env):
    _write(gemini_env, llm_env,
           {"llm_provider": "freellmapi", "llm_api_key": "k",
            "llm_gemini_fallback": False})
    assert gemini.freellmapi_enabled() is False


# ── It is last ───────────────────────────────────────────────────────────────

def test_the_free_rung_sits_at_the_bottom_of_every_ladder():
    for tier, ladder in gemini._LADDERS.items():
        assert ladder[-1] == gemini.FREELLMAPI, tier
        assert ladder.count(gemini.FREELLMAPI) == 1, tier


def test_gemini_is_tried_first(gemini_env, llm_env, monkeypatch):
    """A healthy Gemini must answer; the free router is never consulted."""
    _write(gemini_env, llm_env,
           {"llm_provider": "freellmapi", "llm_api_key": "k"})

    called = []
    monkeypatch.setattr(lc, "call_llm",
                        lambda *a, **kw: called.append(1) or {"content": "free"})

    class _Ok:
        text = "gemini answered"

    monkeypatch.setattr(gemini, "client",
                        lambda **kw: type("C", (), {
                            "models": type("M", (), {
                                "generate_content": lambda self, **k: _Ok()})()})())

    out = gemini.call("hello", tier=gemini.FAST)

    assert out.text == "gemini answered"
    assert called == []


def test_the_free_rung_answers_when_every_gemini_rung_is_exhausted(
        gemini_env, llm_env, monkeypatch):
    _write(gemini_env, llm_env,
           {"llm_provider": "freellmapi", "llm_api_key": "k"})

    seen = {}

    def fake_call_llm(messages, tools=None, timeout=120):
        seen["messages"] = messages
        return {"content": "answered by the free pool", "tool_calls": []}

    monkeypatch.setattr(lc, "call_llm", fake_call_llm)

    out = gemini.call("Summarise this document.", tier=gemini.SMART)

    assert out is not None
    assert out.text == "answered by the free pool"
    # The one-shot persona preamble is still applied — without it a chat model
    # answers "Understood." instead of producing the value that was asked for.
    assert seen["messages"][0]["role"] == "system"
    assert "data-processing function" in seen["messages"][0]["content"]


def test_no_gemini_key_at_all_still_gets_an_answer(gemini_env, llm_env, monkeypatch):
    """An install pointed entirely at a local router has no Gemini key, and
    that is not a reason for every side call to return nothing."""
    _write(gemini_env, llm_env,
           {"gemini_api_key": "", "llm_provider": "freellmapi",
            "llm_api_key": "k"})
    gemini._cached_key = None

    monkeypatch.setattr(lc, "call_llm",
                        lambda *a, **kw: {"content": "from the router",
                                          "tool_calls": []})

    out = gemini.call("hello", tier=gemini.FAST)

    assert out is not None and out.text == "from the router"


def test_a_failing_free_rung_returns_none_like_any_other(gemini_env, llm_env,
                                                         monkeypatch):
    """The ladder's contract: a rung that fails is reported and stepped over."""
    _write(gemini_env, llm_env,
           {"llm_provider": "freellmapi", "llm_api_key": "k"})

    def boom(*a, **kw):
        raise RuntimeError("Cannot reach FreeLLMAPI")

    monkeypatch.setattr(lc, "call_llm", boom)

    assert gemini.call("hello", tier=gemini.FAST) is None


def test_text_helper_returns_the_free_answer(gemini_env, llm_env, monkeypatch):
    """gemini.text() is what most of the app actually calls."""
    _write(gemini_env, llm_env,
           {"llm_provider": "freellmapi", "llm_api_key": "k"})
    monkeypatch.setattr(lc, "call_llm",
                        lambda *a, **kw: {"content": "  three steps  ",
                                          "tool_calls": []})

    assert gemini.text("plan this", tier=gemini.SMART) == "three steps"


def test_as_json_helper_parses_the_free_answer(gemini_env, llm_env, monkeypatch):
    _write(gemini_env, llm_env,
           {"llm_provider": "freellmapi", "llm_api_key": "k"})
    monkeypatch.setattr(lc, "call_llm",
                        lambda *a, **kw: {"content": '```json\n{"steps": [1, 2]}\n```',
                                          "tool_calls": []})

    assert gemini.as_json("plan", tier=gemini.SMART) == {"steps": [1, 2]}


# ── Message shaping ──────────────────────────────────────────────────────────

def test_a_plain_string_becomes_a_plain_user_message():
    msgs = gemini._to_openai_messages("hello", "be brief")
    assert msgs == [{"role": "system", "content": "be brief"},
                    {"role": "user", "content": "hello"}]


def test_an_image_becomes_an_image_url_block():
    """FreeLLMAPI documents standard OpenAI image_url blocks and then routes
    the request to a vision-capable model on its own."""
    msgs = gemini._to_openai_messages(
        ["what is on this screen?", _FakeParts(b"\x00\x01")], "")
    blocks = msgs[-1]["content"]
    assert blocks[0] == {"type": "text", "text": "what is on this screen?"}
    assert blocks[1]["type"] == "image_url"
    assert blocks[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_no_system_message_when_none_is_given():
    msgs = gemini._to_openai_messages("hi", "")
    assert [m["role"] for m in msgs] == ["user"]
