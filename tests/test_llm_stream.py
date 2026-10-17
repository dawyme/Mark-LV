"""Streaming: SSE parsing, sentence splitting, and streamed tool-call assembly.

Mark-LV's callers do not care which provider answered — they want complete
sentences as they arrive (so TTS can start before the model has finished
thinking) and one "done" event carrying the tool calls. That contract is what
these tests pin, for every provider.
"""
import json

import pytest

from core import llm_client as lc
from tests.conftest import FakeResponse


def _sse(*chunks, done=True):
    """Build the SSE lines a server would send, plus the [DONE] terminator."""
    lines = []
    for chunk in chunks:
        lines.append("data: " + json.dumps(chunk))
    if done:
        lines.append("data: [DONE]")
    return lines


def _delta(text=None, tool_calls=None, finish=None):
    d = {}
    if text is not None:
        d["content"] = text
    if tool_calls is not None:
        d["tool_calls"] = tool_calls
    return {"choices": [{"delta": d, "finish_reason": finish}]}


# ── Sentence streaming ───────────────────────────────────────────────────────

def test_sentences_are_yielded_as_they_complete(config_file, queue):
    queue(FakeResponse(200, lines=_sse(
        _delta("Hello there. "),
        _delta("How are "),
        _delta("you today?"),
    )))
    config_file({"llm_provider": "freellmapi", "llm_api_key": "k"})

    events = list(lc.call_llm_stream([{"role": "user", "content": "hi"}]))

    sentences = [e["text"] for e in events if e["type"] == "sentence"]
    assert sentences == ["Hello there.", "How are you today?"]
    assert events[-1]["type"] == "done"
    assert events[-1]["content"] == "Hello there. How are you today?"
    assert events[-1]["tool_calls"] == []


def test_trailing_text_without_a_final_period_is_flushed(config_file, queue):
    queue(FakeResponse(200, lines=_sse(_delta("no terminator here"))))
    config_file({"llm_provider": "openai"})

    events = list(lc.call_llm_stream([{"role": "user", "content": "hi"}]))

    assert [e["text"] for e in events if e["type"] == "sentence"] == ["no terminator here"]


def test_decimals_are_not_treated_as_sentence_ends(config_file, queue):
    queue(FakeResponse(200, lines=_sse(_delta("It costs 3.5 euros. Fine."))))
    config_file({"llm_provider": "openai"})

    events = list(lc.call_llm_stream([{"role": "user", "content": "hi"}]))

    assert [e["text"] for e in events if e["type"] == "sentence"] == [
        "It costs 3.5 euros.", "Fine."]


def test_garbage_lines_are_skipped_not_fatal(config_file, queue):
    queue(FakeResponse(200, lines=[
        ": keep-alive comment",
        "event: ping",
        "data: {not json",
        "data: " + json.dumps(_delta("Recovered. ")),
        "data: [DONE]",
    ]))
    config_file({"llm_provider": "freellmapi", "llm_api_key": "k"})

    events = list(lc.call_llm_stream([{"role": "user", "content": "hi"}]))

    assert events[0] == {"type": "sentence", "text": "Recovered."}
    assert events[-1]["type"] == "done"


# ── Streamed tool calls ──────────────────────────────────────────────────────

def test_streamed_tool_call_fragments_are_assembled(config_file, queue):
    """FreeLLMAPI documents tool calls arriving as delta.tool_calls chunks
    closed by finish_reason "tool_calls". The fragments have to be stitched
    back into one call, with the argument string parsed into a dict."""
    queue(FakeResponse(200, lines=_sse(
        _delta(tool_calls=[{"index": 0, "id": "call_9",
                            "function": {"name": "get_weather", "arguments": ""}}]),
        _delta(tool_calls=[{"index": 0,
                            "function": {"name": "", "arguments": '{"ci'}}]),
        _delta(tool_calls=[{"index": 0,
                            "function": {"name": "", "arguments": 'ty": "Lagos"}'}}]),
        _delta(finish="tool_calls"),
    )))
    config_file({"llm_provider": "freellmapi", "llm_api_key": "k"})

    events = list(lc.call_llm_stream([{"role": "user", "content": "weather?"}],
                                     tools=[{"type": "function"}]))

    done = events[-1]
    assert done["type"] == "done"
    assert done["content"] == ""
    assert done["tool_calls"] == [{
        "id": "call_9",
        "function": {"name": "get_weather", "arguments": {"city": "Lagos"}},
    }]


def test_two_parallel_tool_calls_keep_their_order(config_file, queue):
    queue(FakeResponse(200, lines=_sse(
        _delta(tool_calls=[{"index": 1, "id": "b",
                            "function": {"name": "second", "arguments": "{}"}}]),
        _delta(tool_calls=[{"index": 0, "id": "a",
                            "function": {"name": "first", "arguments": "{}"}}]),
        _delta(finish="tool_calls"),
    )))
    config_file({"llm_provider": "freellmapi", "llm_api_key": "k"})

    done = list(lc.call_llm_stream([{"role": "user", "content": "x"}],
                                   tools=[{}]))[-1]

    assert [t["function"]["name"] for t in done["tool_calls"]] == ["first", "second"]


def test_streamed_malformed_arguments_survive_as_text(config_file, queue):
    queue(FakeResponse(200, lines=_sse(
        _delta(tool_calls=[{"index": 0, "id": "c",
                            "function": {"name": "reminder", "arguments": '{"a":'}}]),
        _delta(finish="tool_calls"),
    )))
    config_file({"llm_provider": "freellmapi", "llm_api_key": "k"})

    done = list(lc.call_llm_stream([{"role": "user", "content": "x"}], tools=[{}]))[-1]

    assert done["tool_calls"][0]["function"]["arguments"] == '{"a":'


def test_stream_request_body_is_openai_shaped(config_file, queue):
    rec = queue(FakeResponse(200, lines=_sse(_delta("ok.", finish="stop"))))
    config_file({"llm_provider": "freellmapi", "llm_api_key": "k"})

    list(lc.call_llm_stream([{"role": "user", "content": "hi"}], tools=[{"t": 1}]))

    call = rec.posts[0]
    assert call["url"] == "http://localhost:3001/v1/chat/completions"
    assert call["stream"] is True
    assert call["json"]["stream"] is True
    assert call["json"]["tools"] == [{"t": 1}]
    assert call["headers"]["Authorization"] == "Bearer k"


# ── Ollama streaming is untouched ────────────────────────────────────────────

def test_ollama_ndjson_stream_still_works(config_file, queue):
    # Ollama's native stream: NDJSON, one object per line, tool calls inside
    # `message`, and `done: true` on the last one.
    queue(FakeResponse(200, lines=[
        json.dumps({"message": {"content": "First. "}, "done": False}),
        json.dumps({"message": {"content": "Second.",
                                "tool_calls": [{"function": {"name": "x",
                                                              "arguments": {}}}]},
                    "done": True}),
    ]))
    config_file({"llm_provider": "ollama"})

    events = list(lc.call_llm_stream([{"role": "user", "content": "hi"}]))

    assert [e["text"] for e in events if e["type"] == "sentence"] == ["First.", "Second."]
    assert events[-1]["tool_calls"] == [{"function": {"name": "x", "arguments": {}}}]
    assert events[-1]["content"] == "First. Second."


def test_ollama_stream_posts_to_the_native_endpoint(config_file, queue):
    rec = queue(FakeResponse(200, lines=[
        json.dumps({"message": {"content": "ok."}, "done": True})]))
    config_file({"llm_provider": "ollama"})

    list(lc.call_llm_stream([{"role": "user", "content": "hi"}]))

    body = rec.posts[0]["json"]
    assert rec.posts[0]["url"].endswith("/api/chat")
    assert body["stream"] is True
    assert body["keep_alive"] == -1
    assert "max_tokens" not in body
