#!/usr/bin/env python3
"""
Wire-contract verification against a REAL HTTP server — no mocks, no network.

    python tests/compat_server_check.py

WHAT THIS IS
    A local HTTP server, written here with nothing but the standard library,
    that implements the FreeLLMAPI surface exactly as the router's own API
    reference documents it:

        POST /v1/chat/completions   OpenAI request/response shape
        GET  /v1/models             catalog listing
        Authorization: Bearer …     401 {"error": {"message", "type"}} without it
        429 + Retry-After           rate limiting
        404                         a model that is not in the catalog
        X-Routed-Via                which provider served the call
        SSE with delta.tool_calls   streaming, tool calls closed by
                                    finish_reason "tool_calls"

    Then it points the SHIPPED core/llm_client.py at it over a real socket and
    checks what comes back.

WHAT THIS IS NOT
    It is not the FreeLLMAPI router. It is a server that speaks the same
    documented contract, which proves Mark-LV's side of the wire is correct —
    real bytes, real status codes, real SSE — without proving that a given
    router deployment is healthy. For that, run tests/live_freellmapi_check.py
    against a real router. Both scripts exist because they answer different
    questions, and saying "it works" on the strength of only one of them is
    how an unverified claim gets made.

Exits 0 only when every check passed.
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

API_KEY   = "freellmapi-test-unified-key"
CATALOG   = ["auto", "fusion", "auto:fast", "gemini-2.5-flash", "llama-3.3-70b"]
_rlim_hit = {"n": 0}


class Handler(BaseHTTPRequestHandler):
    """The documented surface, and nothing else."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def handle_error(self, request, client_address):
        """A client that hangs up mid-response is normal here, not a fault.

        The streaming checks close the connection as soon as the 'done' event
        arrives, which is what a real client does; the default handler prints a
        BrokenPipeError traceback for it and buries the report.
        """
        pass

    # -- helpers --
    def _authed(self) -> bool:
        return self.headers.get("Authorization") == f"Bearer {API_KEY}"

    def _json(self, status, payload, extra_headers=None):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, message, typ="invalid_request_error",
               extra_headers=None):
        self._json(status, {"error": {"message": message, "type": typ}},
                   extra_headers)

    # -- routes --
    def do_GET(self):
        if self.path.split("?")[0] != "/v1/models":
            return self._error(404, "no such endpoint")
        if not self._authed():
            return self._error(401, "Invalid API key", "authentication_error")
        return self._json(200, {"object": "list",
                                "data": [{"id": m, "object": "model"} for m in CATALOG]},
                          {"X-Routed-Via": "router/catalog"})

    def do_POST(self):
        if self.path.split("?")[0] != "/v1/chat/completions":
            return self._error(404, "no such endpoint")
        if not self._authed():
            return self._error(401, "Invalid API key", "authentication_error")

        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return self._error(400, "request body is not JSON")

        # Ollama-only fields must never arrive here; if they do, the client is
        # speaking the wrong dialect and this is the place to find out.
        for foreign in ("keep_alive", "options", "num_predict"):
            if foreign in body:
                return self._error(400, f"unexpected Ollama field '{foreign}'")

        model = body.get("model") or ""
        if model not in CATALOG and not model.startswith("auto:"):
            return self._error(404, f"model '{model}' not found")

        text = ""
        for message in body.get("messages") or []:
            content = message.get("content")
            if isinstance(content, str):
                text += content

        if "RATELIMIT" in text:
            _rlim_hit["n"] += 1
            if _rlim_hit["n"] == 1:
                return self._error(429, "every provider is rate limited",
                                   "rate_limit_error", {"Retry-After": "1"})

        tools = body.get("tools") or []
        if tools:
            return self._tool_reply(body)
        if body.get("stream"):
            return self._stream_reply(text)
        return self._json(200, {
            "id": "chatcmpl-1", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant",
                                     "content": f"You said: {text[:60]}"}}],
        }, {"X-Routed-Via": "google/gemini-2.5-flash"})

    def _tool_reply(self, body):
        if body.get("stream"):
            chunks = [
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "id": "call_abc",
                     "function": {"name": "get_weather", "arguments": ""}}]},
                    "finish_reason": None}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "function": {"name": "", "arguments": '{"ci'}}]},
                    "finish_reason": None}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "function": {"name": "", "arguments": 'ty": "Karachi"}'}}]},
                    "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            ]
            return self._sse(chunks)
        return self._json(200, {
            "id": "chatcmpl-2", "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "tool_calls",
                         "message": {"role": "assistant", "content": None,
                                     "tool_calls": [{
                                         "id": "call_abc", "type": "function",
                                         "function": {
                                             "name": "get_weather",
                                             "arguments": '{"city": "Karachi"}'}}]}}],
        }, {"X-Routed-Via": "groq/llama-3.3-70b"})

    def _stream_reply(self, text):
        words = f"First sentence about {text[:20]!r}. Second sentence here. Tail".split(" ")
        chunks = [{"choices": [{"delta": {"content": w + " "}, "finish_reason": None}]}
                  for w in words]
        chunks.append({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        return self._sse(chunks)

    def _sse(self, chunks):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("X-Routed-Via", "google/gemini-2.5-flash")
        self.end_headers()
        for chunk in chunks:
            payload = ("data: " + json.dumps(chunk) + "\n\n").encode("utf-8")
            self.wfile.write(b"%x\r\n%s\r\n" % (len(payload), payload))
        tail = b"data: [DONE]\n\n"
        self.wfile.write(b"%x\r\n%s\r\n" % (len(tail), tail))
        self.wfile.write(b"0\r\n\r\n")


def start_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


CHECKS = []


def check(name):
    def wrap(fn):
        CHECKS.append((name, fn))
        return fn
    return wrap


def run_all(base_url):
    import os
    from core import llm_client as lc

    os.environ["MARK_LLM_PROVIDER"] = "freellmapi"
    os.environ["MARK_LLM_URL"]      = base_url
    os.environ["MARK_LLM_MODEL"]    = "auto"
    os.environ["FREELLMAPI_API_KEY"] = API_KEY

    @check("configuration resolves to the test server")
    def c0():
        assert lc.get_llm_provider() == "freellmapi", lc.get_llm_provider()
        assert lc.get_llm_settings() == (base_url, "auto"), lc.get_llm_settings()
        assert lc.get_api_key() == API_KEY
        return f"{lc.provider_label()}"

    @check("GET /v1/models lists the catalog")
    def c1():
        models = lc.list_models(timeout=10)
        assert models == CATALOG, models
        return f"{len(models)} models"

    @check("missing key is a 401 the user can act on")
    def c2():
        os.environ["FREELLMAPI_API_KEY"] = "wrong"
        try:
            lc.call_llm_text("hi", timeout=10)
            return "FAIL: accepted a bad key"
        except RuntimeError as e:
            assert "401" in str(e) and "llm_api_key" in str(e), str(e)
            return str(e)[:80]
        finally:
            os.environ["FREELLMAPI_API_KEY"] = API_KEY

    @check("an unknown model is a 404, not a guess")
    def c3():
        os.environ["MARK_LLM_MODEL"] = "no-such-model"
        try:
            lc.call_llm_text("hi", timeout=10)
            return "FAIL: accepted an unknown model"
        except RuntimeError as e:
            assert "404" in str(e), str(e)
            return str(e)[:80]
        finally:
            os.environ["MARK_LLM_MODEL"] = "auto"

    @check("429 is retried after Retry-After and then succeeds")
    def c4():
        out = lc.call_llm([{"role": "user", "content": "RATELIMIT please"}],
                          timeout=15)
        assert "You said" in out["content"], out
        return out["content"][:60]

    @check("non-streaming chat returns content")
    def c5():
        out = lc.call_llm([{"role": "user", "content": "hello world"}], timeout=10)
        assert out["content"] == "You said: hello world", out
        assert out["tool_calls"] == []
        return out["content"]

    @check("streaming yields sentences then a done event")
    def c6():
        events = list(lc.call_llm_stream(
            [{"role": "user", "content": "the weather"}], timeout=15))
        sentences = [e["text"] for e in events if e["type"] == "sentence"]
        assert len(sentences) >= 2, sentences
        assert events[-1]["type"] == "done"
        assert "First sentence" in events[-1]["content"]
        return f"{len(sentences)} sentences: {sentences[0][:40]!r}"

    @check("tool calls come back parsed into a dict")
    def c7():
        tools = [{"type": "function", "function": {
            "name": "get_weather",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}},
                           "required": ["city"]}}}]
        out = lc.call_llm([{"role": "user", "content": "weather in Karachi?"}],
                          tools=tools, timeout=10)
        tc = out["tool_calls"]
        assert len(tc) == 1, tc
        assert tc[0]["function"]["name"] == "get_weather"
        assert tc[0]["function"]["arguments"] == {"city": "Karachi"}
        return f"{tc[0]['function']['name']}({tc[0]['function']['arguments']})"

    @check("streamed tool-call fragments are stitched back together")
    def c8():
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        done = list(lc.call_llm_stream(
            [{"role": "user", "content": "weather?"}], tools=tools,
            timeout=15))[-1]
        assert done["tool_calls"][0]["function"]["arguments"] == {"city": "Karachi"}, done
        return "arguments reassembled across three SSE chunks"

    @check("call_llm_text speaks OpenAI, never Ollama")
    def c9():
        # The server rejects keep_alive/options/num_predict with a 400, so a
        # payload in the wrong dialect cannot pass this check.
        text = lc.call_llm_text("ping", system="Reply with one word.", timeout=10)
        assert "ping" in text, text
        return text

    results = []
    for name, fn in CHECKS:
        try:
            detail = fn()
            ok = not str(detail).startswith("FAIL")
        except AssertionError as e:
            ok, detail = False, f"assertion failed: {e}"
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {e}"
        results.append((name, ok, str(detail)))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}"
              + (f"  — {detail}" if ok and detail else f"\n          {detail}" if not ok else ""))
    return results


def main():
    print("FreeLLMAPI wire-contract verification against a local HTTP server.")
    print("Real sockets, real SSE, real status codes. Not a mock, and not the")
    print("router itself — for that, run tests/live_freellmapi_check.py.\n")

    server, port = start_server()
    base = f"http://127.0.0.1:{port}/v1"
    try:
        results = run_all(base)
    finally:
        server.shutdown()

    passed = sum(1 for _n, ok, _d in results if ok)
    print("\n" + "=" * 70)
    print(f"{passed}/{len(results)} checks passed against {base}")
    print("=" * 70)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
