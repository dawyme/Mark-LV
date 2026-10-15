"""Shared fixtures for the Mark-LV test suite.

Everything here is offline by design. The LLM tests drive core/llm_client.py
through a fake `requests`, so they assert the bytes that would go on the wire
without needing an Ollama, an LM Studio or a FreeLLMAPI router to be running —
and without ever touching the network, which is the only way a test that runs
on a machine with no keys still means something.

Run with:  python -m pytest            (or:  python -m pytest -q tests)
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── A stand-in for requests.Response ────────────────────────────────────────

class FakeResponse:
    """Enough of requests.Response for the client under test.

    `lines` turns this into a streaming response: iter_lines() yields them the
    way the real one yields SSE chunks, which is what the streaming tests feed
    in.
    """

    def __init__(self, status_code=200, payload=None, text="", headers=None,
                 lines=None, raise_on_json=False):
        self.status_code = status_code
        self._payload    = payload
        self.text        = text if text else (
            json.dumps(payload) if payload is not None else "")
        self.headers     = headers or {}
        self._lines      = lines or []
        self._raise_json = raise_on_json

    def json(self):
        if self._raise_json:
            raise ValueError("no JSON in body")
        if self._payload is None:
            raise ValueError("no JSON in body")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            err = requests.exceptions.HTTPError(
                f"{self.status_code} for url", response=self)
            raise err

    def iter_lines(self):
        for line in self._lines:
            yield line.encode("utf-8") if isinstance(line, str) else line

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Recorder:
    """Captures every call the client makes, and hands back queued responses.

    `queue` is consumed in order; the last response repeats if the client makes
    more calls than were queued, so a retry test can queue one 429 and one 200
    without caring how many attempts follow.
    """

    def __init__(self, queue=None):
        self.queue = list(queue or [])
        self.posts = []      # (url, json, headers, kwargs)
        self.gets  = []      # (url, kwargs)

    # -- responses --
    def _next(self):
        if not self.queue:
            return FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]})
        return self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]

    # -- requests.post / requests.get --
    def post(self, url, json=None, headers=None, timeout=None, stream=False, **kw):
        self.posts.append({"url": url, "json": json, "headers": headers or {},
                           "timeout": timeout, "stream": stream, **kw})
        return self._next()

    def get(self, url, headers=None, timeout=None, **kw):
        self.gets.append({"url": url, "headers": headers or {}, "timeout": timeout, **kw})
        return self._next()


@pytest.fixture
def recorder(monkeypatch):
    """Patch requests.post/get inside core.llm_client and return the recorder."""
    from core import llm_client

    rec = Recorder()
    monkeypatch.setattr(llm_client.requests, "post", rec.post)
    monkeypatch.setattr(llm_client.requests, "get", rec.get)
    return rec


@pytest.fixture
def queue(monkeypatch):
    """Patch requests with a caller-supplied response queue."""
    from core import llm_client

    def _apply(*responses):
        rec = Recorder(list(responses))
        monkeypatch.setattr(llm_client.requests, "post", rec.post)
        monkeypatch.setattr(llm_client.requests, "get", rec.get)
        return rec
    return _apply


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    """Point core.llm_client at a scratch config file, and clear LLM env vars.

    Returns a callable that writes a dict into that file, so a test can set the
    provider exactly the way a user would.
    """
    from core import llm_client

    path = tmp_path / "api_keys.json"
    monkeypatch.setattr(llm_client, "CONFIG_PATH", path)
    for var in ("MARK_LLM_PROVIDER", "MARK_LLM_URL", "MARK_LLM_MODEL",
                "MARK_LLM_API_KEY", "MARK_LLM_MAX_TOKENS",
                "FREELLMAPI_API_KEY", "FREELLMAPI_BASE_URL", "FREELLMAPI_MODEL",
                "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        monkeypatch.delenv(var, raising=False)

    def _write(data: dict):
        path.write_text(json.dumps(data), encoding="utf-8")
        return data
    return _write
