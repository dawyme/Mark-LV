"""A guard for the one thing this integration must not touch: the voice.

Mark-LV's voice is Gemini Live — a native-audio model, a prebuilt voice chosen
in settings, and an audio pipeline in main.py that opens the microphone, plays
PCM to the speakers and derives mouth shapes from the samples. Adding a second
language model to the app creates exactly one risk that matters: that model
finding its way into the conversation, and the assistant starting to speak in a
voice nobody chose.

main.py cannot be imported in a test (it needs PyQt6, sounddevice and the
Gemini SDK), so these checks read the shipped source with `ast` instead. That
is still a check on the real file — if the speech config moves, or the LLM
client gets imported into the Live loop, the assertion fails on the code as it
sits in the repository.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _source(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def _tree(rel):
    return ast.parse(_source(rel))


def _imported_names(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(f"{node.module}.{a.name}" for a in node.names)
    return names


def _function_source(tree, qualname):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == qualname:
            return ast.get_source_segment(_source_of(tree), node) or ""
    return ""


_CACHE = {}


def _source_of(tree):
    key = id(tree)
    if key not in _CACHE:
        raise KeyError("use _function_source_from(rel, name)")
    return _CACHE[key]


def _function_source_from(rel, name):
    src = _source(rel)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == name:
            return ast.get_source_segment(src, node) or ""
    return ""


# ── The Live session is still the conversation ───────────────────────────────

def test_the_speech_config_still_uses_the_chosen_prebuilt_voice():
    body = _function_source_from("main.py", "_build_config")

    assert body, "_build_config not found in main.py"
    assert "speech_config=types.SpeechConfig" in body
    assert "prebuilt_voice_config=types.PrebuiltVoiceConfig" in body
    assert "voice_name=get_voice()" in body
    # The conversation is spoken by the model, not synthesised from its text.
    assert 'response_modalities=["AUDIO"]' in body


def test_no_second_voice_provider_was_introduced():
    """Nothing in the Live loop may synthesise speech. The assistant's voice is
    the Live model's own audio, so any TTS engine in the conversation path is
    a different voice wearing the same name."""
    imports = _imported_names(_tree("main.py"))

    assert not any("tts" in name for name in imports)
    assert not any("llm_client" in name for name in imports)


def test_main_never_imports_the_llm_client():
    """The free provider is a reasoning fallback for one-shot calls. It has no
    business in the module that owns the microphone and the speaker."""
    src = _source("main.py")

    assert "llm_client" not in src


def test_the_llm_client_touches_no_audio_at_all():
    src = _source("core/llm_client.py")

    for forbidden in ("sounddevice", "import numpy", "pyaudio", "edge_tts",
                      "elevenlabs", "kokoro", "speak(", "response_modalities"):
        assert forbidden not in src, forbidden


# ── The voice picker is unchanged ────────────────────────────────────────────

def test_the_voice_list_and_default_are_untouched():
    from memory import config_manager

    assert config_manager.AVAILABLE_VOICES == ["Charon", "Puck", "Kore",
                                               "Fenrir", "Aoede"]
    assert config_manager.DEFAULT_VOICE == "Charon"


def test_an_unknown_voice_still_collapses_to_the_default(tmp_path, monkeypatch):
    """A bad value must never reach the API and break the session."""
    from memory import config_manager

    cfg = tmp_path / "api_keys.json"
    cfg.write_text('{"voice_name": "SomeFreeTT SVoice"}', encoding="utf-8")
    monkeypatch.setattr(config_manager, "CONFIG_FILE", cfg)

    assert config_manager.get_voice() == "Charon"


# ── The conversation model is still Gemini Live ──────────────────────────────

def test_the_live_models_are_still_live_models():
    from core import gemini

    assert gemini.LIVE_MODELS[0] == "models/gemini-3.1-flash-live-preview"
    assert all(m.startswith("models/") for m in gemini.LIVE_MODELS)


def test_the_free_rung_is_not_a_live_model():
    """The conversation's model list and the one-shot ladder are different
    things; the free rung belongs only to the latter."""
    from core import gemini

    assert gemini.FREELLMAPI not in gemini.LIVE_MODELS
    assert gemini.LIVE not in gemini._LADDERS[gemini.SEARCH]


# ── New capabilities do not speak ────────────────────────────────────────────

@pytest.mark.parametrize("rel", ["actions/task_agent.py", "actions/research.py",
                                 "core/task_engine.py"])
def test_new_capabilities_never_call_speak(rel):
    """They return text and let main.py's existing pipeline decide how it is
    said. An action that calls speak() itself produces a second voice with no
    lip-sync, no echo guard and no interrupt handling behind it."""
    src = _source(rel)

    assert "speak(" not in src


def test_the_prompt_was_not_rewritten():
    """The personality lives in core/prompt.txt. Nothing in this integration
    needs a word of it, and a system prompt quietly growing is how an assistant
    starts sounding like a different one."""
    src = _source("core/prompt.txt")

    for forbidden in ("freellmapi", "FreeLLMAPI", "llm_client", "ollama"):
        assert forbidden.lower() not in src.lower(), forbidden
