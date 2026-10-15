#!/usr/bin/env python3
"""
Live verification against a real FreeLLMAPI router — NOT part of the test suite.

    python tests/live_freellmapi_check.py

Why this exists separately: `python -m pytest` must pass on a machine with no
API keys and no network, so every test in tests/ drives core/llm_client.py
through a fake transport. That proves the request format, the parsing and the
error handling; it does not prove that a real router accepts the request. This
script closes that gap, and it is the thing to run after changing anything in
core/llm_client.py before claiming the integration works.

It exercises the SHIPPED code paths — llm_client.list_models(), call_llm(),
call_llm_text() and call_llm_stream() — not a parallel reimplementation, so a
pass here means the real functions spoke to a real server.

Set up first (see docs/FREELLMAPI.md):
    1. run the router:  https://github.com/tashfeenahmed/freellmapi
    2. add at least one free provider key on its Keys page
    3. put the unified key in config/api_keys.json as "llm_api_key"
       (or export FREELLMAPI_API_KEY)

Exits 0 only when every step passed.
"""
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import llm_client as lc          # noqa: E402

WEATHER_TOOL = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string",
                                    "description": "City name"}},
            "required": ["city"],
        },
    },
}]

_results = []


def step(name):
    def wrap(fn):
        def runner():
            print(f"\n── {name} " + "─" * max(0, 60 - len(name)))
            try:
                ok, detail = fn()
            except Exception as e:
                traceback.print_exc()
                ok, detail = False, f"{type(e).__name__}: {e}"
            print(("   PASS  " if ok else "   FAIL  ") + str(detail))
            _results.append((name, ok, str(detail)))
        runner.__wrapped_name__ = name
        return runner
    return wrap


@step("0. configuration")
def check_config():
    info = lc.diagnose()
    if info["provider"] != lc.FREELLMAPI:
        return False, (f"llm_provider is '{info['provider']}', not 'freellmapi'. "
                       "Set it in config/api_keys.json.")
    if not info["api_key_set"]:
        return False, ("no API key configured — set \"llm_api_key\" in "
                       "config/api_keys.json or export FREELLMAPI_API_KEY")
    if not info["reachable"]:
        return False, f"router not reachable at {info['base_url']} — {info['error']}"
    return True, (f"{info['base_url']}  model={info['model']}  "
                  f"reachable, {len(info['models'])} models listed")


@step("1. GET /v1/models")
def check_models():
    models = lc.list_models(timeout=15)
    if not models:
        return False, "no models listed — is a provider key added on the Keys page?"
    shown = ", ".join(models[:8])
    return True, f"{len(models)} models — {shown}{'…' if len(models) > 8 else ''}"


@step("2. POST /v1/chat/completions (non-streaming, via call_llm)")
def check_chat():
    out = lc.call_llm([{"role": "system", "content": "Answer in one short sentence."},
                       {"role": "user", "content": "Name one planet in this solar system."}],
                      timeout=90)
    content = out.get("content") or ""
    if not content:
        return False, "the router answered with no content"
    return True, f"{len(content)} chars — {content[:90]!r}"


@step("3. call_llm_text (the path that used to be Ollama-only)")
def check_text():
    text = lc.call_llm_text("Reply with exactly the word: pong",
                            system="You output one word and nothing else.",
                            timeout=90)
    if not text:
        return False, "empty reply"
    return True, f"{len(text)} chars — {text[:90]!r}"


@step("4. POST /v1/chat/completions (stream: true, via call_llm_stream)")
def check_stream():
    sentences, done = [], None
    for event in lc.call_llm_stream(
            [{"role": "user", "content": "Count from one to five in full sentences."}],
            timeout=90):
        if event["type"] == "sentence":
            sentences.append(event["text"])
        else:
            done = event
    if done is None:
        return False, "the stream ended without a 'done' event"
    if not sentences:
        return False, "the stream produced no sentences"
    return True, (f"{len(sentences)} sentence event(s), first: "
                  f"{sentences[0][:60]!r}; content {len(done['content'])} chars")


@step("5. tool calling (tools + tool_choice)")
def check_tools():
    out = lc.call_llm([{"role": "user", "content": "What is the weather in Karachi?"}],
                      tools=WEATHER_TOOL, timeout=90)
    calls = out.get("tool_calls") or []
    if not calls:
        # Not every model on a free tier supports tool calling, and the router
        # may have served the request from one that does not. That is a real
        # finding, so it is reported rather than passed.
        return False, ("no tool call returned — the model that served this may "
                       f"not support tool calling. Content: {out.get('content', '')[:80]!r}")
    first = calls[0]
    args = first["function"]["arguments"]
    if not isinstance(args, dict):
        return False, f"arguments did not parse to a dict: {args!r}"
    return True, (f"{first['function']['name']}({args}) — id={first['id'][:16]}")


@step("6. a malformed tool argument survives as data")
def check_malformed():
    """Fed a model that emits broken JSON, the client must hand back the raw
    string rather than raising — main.py's dispatcher copes with that shape."""
    parsed = lc._parse_tool_arguments('{"city": "Karachi",')
    if parsed != '{"city": "Karachi",':
        return False, f"unexpected handling: {parsed!r}"
    return True, "broken JSON is returned as-is, not raised"


@step("7. an unusable model is reported, not guessed at")
def check_bad_model():
    import os
    previous = os.environ.get("MARK_LLM_MODEL")
    os.environ["MARK_LLM_MODEL"] = "definitely-not-a-real-model-xyz"
    try:
        lc.call_llm_text("hi", timeout=60)
        return False, "a nonexistent model was accepted — unexpected"
    except RuntimeError as e:
        msg = str(e)
        if "404" in msg or "not" in msg.lower() or "400" in msg:
            return True, f"rejected as expected: {msg[:110]}"
        return False, f"failed, but not in a way the user can act on: {msg[:110]}"
    finally:
        if previous is None:
            os.environ.pop("MARK_LLM_MODEL", None)
        else:
            os.environ["MARK_LLM_MODEL"] = previous


def main():
    print("FreeLLMAPI live verification — real requests, real router.")
    print("Nothing here is mocked. See docs/FREELLMAPI.md for the setup.")

    check_config()
    configured = _results[-1][1]

    # Everything after step 0 needs a reachable router. Running them anyway
    # produces eight connection errors and a report that looks like eight
    # separate problems, and — worse — a step can "pass" for the wrong reason
    # when the only thing it observed was a refusal to connect.
    live_steps = (check_models, check_chat, check_text, check_stream,
                  check_tools, check_bad_model)
    offline_steps = (check_malformed,)      # pure parsing, needs no server

    if configured:
        for runner in live_steps:
            runner()
    else:
        for runner in live_steps:
            _results.append((runner.__wrapped_name__, False,
                             "skipped — the router is not reachable"))
    for runner in offline_steps:
        runner()

    passed = sum(1 for _n, ok, _d in _results if ok)
    print("\n" + "=" * 70)
    for name, ok, _detail in _results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("=" * 70)
    print(f"{passed}/{len(_results)} steps passed.")
    if passed != len(_results):
        print("\nNot a verified integration until every step passes.")
        if not configured:
            print("Start the router and set the key first — see docs/FREELLMAPI.md.")
        return 1
    print("\nVerified against a live FreeLLMAPI router.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
