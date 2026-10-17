"""Provider selection and configuration — who answers, at which URL, with which model.

These pin the contract a user configures against, and the precedence between
the config file and the environment. Nothing here reaches the network.
"""
import pytest

from core import llm_client as lc


# ── Selection ────────────────────────────────────────────────────────────────

def test_default_provider_is_ollama(config_file):
    """An untouched config must behave exactly as it did before this change."""
    config_file({})
    assert lc.get_llm_provider() == "ollama"


def test_explicit_providers(config_file):
    for raw, expected in (("ollama", "ollama"),
                          ("openai", "openai"),
                          ("freellmapi", "freellmapi")):
        config_file({"llm_provider": raw})
        assert lc.get_llm_provider() == expected, raw


@pytest.mark.parametrize("alias,expected", [
    ("LMStudio", "openai"), ("lm-studio", "openai"), ("LocalAI", "openai"),
    ("Jan", "openai"), ("llamacpp", "openai"), ("vLLM", "openai"),
    ("FreeLLMAPI", "freellmapi"), ("free-llm-api", "freellmapi"),
    ("free_llm_api", "freellmapi"), ("freellm", "freellmapi"),
    ("FREELLMAPI ", "freellmapi"),
])
def test_provider_aliases(config_file, alias, expected):
    """The spellings people actually type all resolve, and case never matters."""
    config_file({"llm_provider": alias})
    assert lc.get_llm_provider() == expected


def test_unknown_provider_falls_back_to_ollama(config_file):
    """A typo must not leave the assistant with no provider at all."""
    config_file({"llm_provider": "cladu"} )
    assert lc.get_llm_provider() == "ollama"


def test_env_overrides_config(config_file, monkeypatch):
    """An explicit env var beats a stored value, so a deployment can redirect
    the assistant without rewriting the file that also holds the Gemini key."""
    config_file({"llm_provider": "ollama"})
    monkeypatch.setenv("MARK_LLM_PROVIDER", "freellmapi")
    assert lc.get_llm_provider() == "freellmapi"


# ── Defaults per provider ────────────────────────────────────────────────────

def test_ollama_defaults_unchanged(config_file):
    config_file({"llm_provider": "ollama"})
    assert lc.get_llm_settings() == ("http://localhost:11434", "llama3.2")


def test_openai_defaults(config_file):
    config_file({"llm_provider": "openai"})
    assert lc.get_llm_settings() == ("http://localhost:1234/v1", "local-model")


def test_freellmapi_defaults_match_the_documented_contract(config_file):
    """The router's own API reference: base http://localhost:3001/v1, model auto."""
    config_file({"llm_provider": "freellmapi"})
    assert lc.get_llm_settings() == ("http://localhost:3001/v1", "auto")


def test_freellmapi_base_url_gains_the_v1_suffix(config_file):
    """Pasting the dashboard's address bar must still work."""
    config_file({"llm_provider": "freellmapi", "llm_url": "http://localhost:3001"})
    assert lc.get_llm_settings()[0] == "http://localhost:3001/v1"


def test_freellmapi_base_url_keeps_an_explicit_version(config_file):
    config_file({"llm_provider": "freellmapi", "llm_url": "https://router.example/v1/"})
    assert lc.get_llm_settings()[0] == "https://router.example/v1"


def test_ollama_url_never_gains_a_v1_suffix(config_file):
    """Ollama serves /api/*, not /v1 — appending one would break it."""
    config_file({"llm_provider": "ollama", "llm_url": "http://192.168.1.5:11434/"})
    assert lc.get_llm_settings()[0] == "http://192.168.1.5:11434"


def test_remote_hosted_base_url_is_accepted(config_file):
    """A hosted deployment is just a different base URL — nothing hardcodes localhost."""
    config_file({"llm_provider": "freellmapi",
                 "llm_url": "https://api.example.com/v1",
                 "llm_model": "glm-5.2"})
    assert lc.get_llm_settings() == ("https://api.example.com/v1", "glm-5.2")


# ── Keys ─────────────────────────────────────────────────────────────────────

def test_key_from_config(config_file):
    config_file({"llm_provider": "freellmapi", "llm_api_key": "freellmapi-abc123"})
    assert lc.get_api_key() == "freellmapi-abc123"
    assert lc.is_configured() is True


def test_provider_specific_config_key_wins(config_file):
    config_file({"llm_provider": "freellmapi",
                 "llm_api_key": "generic",
                 "freellmapi_api_key": "specific"})
    assert lc.get_api_key() == "specific"


def test_key_from_env(config_file, monkeypatch):
    config_file({"llm_provider": "freellmapi"})
    monkeypatch.setenv("FREELLMAPI_API_KEY", "freellmapi-from-env")
    assert lc.get_api_key() == "freellmapi-from-env"


def test_generic_env_key(config_file, monkeypatch):
    config_file({"llm_provider": "openai"})
    monkeypatch.setenv("MARK_LLM_API_KEY", "sk-something")
    assert lc.get_api_key() == "sk-something"


def test_no_key_means_not_configured_for_a_remote_provider(config_file):
    config_file({"llm_provider": "freellmapi"})
    assert lc.get_api_key() == ""
    assert lc.is_configured() is False


def test_ollama_is_configured_without_a_key(config_file):
    """A local Ollama authenticates with nothing at all."""
    config_file({"llm_provider": "ollama"})
    assert lc.is_configured() is True


def test_provider_label_never_leaks_the_key(config_file):
    config_file({"llm_provider": "freellmapi", "llm_api_key": "freellmapi-secret"})
    label = lc.provider_label()
    assert "freellmapi-secret" not in label
    assert "bearer key" in label


# ── Token ceilings ───────────────────────────────────────────────────────────

def test_max_tokens_defaults(config_file):
    config_file({})
    assert lc.get_max_tokens("chat") == 150
    assert lc.get_max_tokens("text") == 600


def test_max_tokens_from_config(config_file):
    config_file({"llm_max_tokens": 320})
    assert lc.get_max_tokens("chat") == 320
    assert lc.get_max_tokens("text") == 320


def test_max_tokens_garbage_falls_back(config_file):
    config_file({"llm_max_tokens": "lots"})
    assert lc.get_max_tokens("chat") == 150


def test_max_tokens_env_overrides(config_file, monkeypatch):
    config_file({"llm_max_tokens": 320})
    monkeypatch.setenv("MARK_LLM_MAX_TOKENS", "64")
    assert lc.get_max_tokens("chat") == 64
