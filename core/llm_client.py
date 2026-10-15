"""
Local & hosted LLM client for MARK LV.

Supports three backends — selected via  "llm_provider"  in config/api_keys.json
(or the MARK_LLM_PROVIDER environment variable):

  "llm_provider": "ollama"   (default)
        Uses Ollama's native /api/chat endpoint.
        Download: https://ollama.com
        Default port: 11434

  "llm_provider": "openai"
        Uses any OpenAI-compatible server: LM Studio, Jan, LocalAI,
        llama.cpp server, vLLM, etc.
        LM Studio download: https://lmstudio.ai   (default port: 1234)
        Set  "llm_url": "http://localhost:1234"  in config.
        Note: tool-calling support depends on the model; use a model that
        supports function/tool calls (e.g. Qwen2.5, Llama-3.1, Mistral).

  "llm_provider": "freellmapi"
        FreeLLMAPI — the open-source, self-hosted router that puts every free
        LLM tier behind ONE OpenAI-compatible /v1 endpoint and fails over
        between providers when one is rate-limited.
        https://github.com/tashfeenahmed/freellmapi   (default port: 3001)

        THE CONTRACT THIS FILE SPEAKS, taken from the router's own API
        reference (docs/en/api/01-rest-api.md) rather than assumed:

          base URL      http://localhost:3001/v1     (configurable)
          auth          Authorization: Bearer <unified key from the Keys page>
          chat          POST {base}/chat/completions  — OpenAI request/response
          models        GET  {base}/models            (?execution_status=ready)
          model field   "auto" (router picks), "auto:fast" | "auto:smart" |
                        "auto:reliable" | "auto:balanced" | "auto:cheap",
                        "auto:<profile>", "fusion", or a catalog model id
          streaming     supported — SSE, stream: true, tool calls arrive as
                        delta.tool_calls chunks closed by finish_reason
                        "tool_calls"
          tool calling  supported — OpenAI `tools` + `tool_choice` round-trip
          errors        OpenAI envelope {"error": {"message", "type"}};
                        401 authentication_error, 429 rate limit,
                        400 unknown routing profile
          info headers  X-Routed-Via: <platform>/<model>, plus
                        X-Fallback-Attempts / X-Fallback-Trail on a failover

        Because the surface is OpenAI-compatible, this provider shares one code
        path with "openai"; what differs is the defaults, the auth header, the
        401/429 wording, and the X-Routed-Via line it logs. Nothing here talks
        to the microphone or the speaker — the assistant's VOICE stays on
        Gemini Live (see main.py `_build_config` and core/gemini.py).

WHAT THIS CLIENT IS FOR
    It is the *second* brain, not the voice. Gemini Live owns the conversation
    and the audio; this client owns the offline/background reasoning — planning
    a multi-step task, decomposing a research question, reading a document when
    the Gemini text quota has run dry. core/gemini.py can use it as the last
    rung of its ladder, which is how every existing skill inherits it without
    being edited.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Generator

import requests

# Matches a sentence boundary: [.!?] followed by whitespace, or a blank line.
# Avoids splitting on decimals (3.5) because those have no space after the dot.
_SENT_END = re.compile(r'(?<=[.!?])\s+|(?<=\n)\s*\n')

def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR    = get_base_dir()
CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"

# Provider names as they appear in config. Everything else is an alias for one
# of these, resolved in get_llm_provider() so a caller can never receive a
# provider this module does not implement.
OLLAMA     = "ollama"
OPENAI     = "openai"
FREELLMAPI = "freellmapi"
_PROVIDERS = (OLLAMA, OPENAI, FREELLMAPI)

# Anything a user might reasonably type that means "an OpenAI-compatible box".
_OPENAI_ALIASES = ("openai", "lmstudio", "lm-studio", "localai", "jan",
                   "llamacpp", "llama.cpp", "vllm")
# …and the spellings that mean the FreeLLMAPI router.
_FREELLMAPI_ALIASES = ("freellmapi", "free-llm-api", "free_llm_api", "freellm")

_DEFAULTS = {
    OLLAMA: {
        "url":   "http://localhost:11434",
        "model": "llama3.2",
    },
    OPENAI: {
        "url":   "http://localhost:1234",
        "model": "local-model",
    },
    FREELLMAPI: {
        # The router's own documented default. `/v1` is part of the base URL —
        # the client appends only `chat/completions` to whatever is configured.
        "url":   "http://localhost:3001/v1",
        # "auto" is not a model: it tells the router to walk the fallback chain
        # and serve the request from whichever free tier has room. It is the
        # documented default and the one that survives a provider going dark.
        "model": "auto",
    },
}

# Environment variables, checked BEFORE the config file. An explicit env var
# beats a stored value so a deployment can point the assistant at a different
# provider without rewriting a file that also holds the Gemini key — and so a
# test can pin a provider without touching the user's real config.
_ENV_PROVIDER   = "MARK_LLM_PROVIDER"
_ENV_URL        = "MARK_LLM_URL"
_ENV_MODEL      = "MARK_LLM_MODEL"
_ENV_API_KEY    = "MARK_LLM_API_KEY"
_ENV_MAX_TOKENS = "MARK_LLM_MAX_TOKENS"

# Per-provider convenience variables, so a FreeLLMAPI key can live in the
# environment under its own name instead of a generic one.
_PROVIDER_ENV = {
    FREELLMAPI: {"url": "FREELLMAPI_BASE_URL", "model": "FREELLMAPI_MODEL",
                 "api_key": "FREELLMAPI_API_KEY"},
    OPENAI:     {"url": "OPENAI_BASE_URL", "model": "OPENAI_MODEL",
                 "api_key": "OPENAI_API_KEY"},
}

# How many tokens a reply may use. Chat replies are short by design — this is a
# voice assistant, and a 150-token cap is roughly three spoken sentences. The
# text-only path (planning, code, documents) needs more room.
_MAX_TOKENS_CHAT = 150
_MAX_TOKENS_TEXT = 600

# Rate limiting. FreeLLMAPI answers 429 when every free tier in the pool is
# momentarily exhausted, and it says how long to wait in Retry-After. Retrying
# once after that header is the difference between a failed task and a slightly
# late one; retrying in a tight loop would only make the cooldown longer.
_RETRY_LIMIT_ATTEMPTS = 2
_RETRY_BACKOFF        = (1.0, 3.0)
_RETRY_MAX_WAIT       = 20.0


# ── Configuration ─────────────────────────────────────────────────────────────

def _load_config() -> dict:
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def _norm_provider(raw: str) -> str:
    """Map anything a user typed onto a provider this module implements."""
    key = (raw or "").strip().lower().replace("_", "-")
    if key in _FREELLMAPI_ALIASES or key.replace("-", "_") in _FREELLMAPI_ALIASES:
        return FREELLMAPI
    if key in _OPENAI_ALIASES or key.replace("-", "_") in _OPENAI_ALIASES:
        return OPENAI
    return OLLAMA


def get_llm_provider() -> str:
    """Returns 'ollama', 'openai' or 'freellmapi'.

    'openai' covers LM Studio, LocalAI, Jan, vLLM — anything OpenAI-compatible.
    An unrecognised value falls back to Ollama, exactly as before, so a typo in
    the config cannot leave the assistant with no provider at all.
    """
    raw = _env(_ENV_PROVIDER) or _load_config().get("llm_provider", "")
    return _norm_provider(raw or OLLAMA)


def _normalize_base(url: str) -> str:
    """Trim a trailing slash, and add the `/v1` an OpenAI client expects.

    FreeLLMAPI's documented base URL already ends in `/v1`, and so does every
    other OpenAI-compatible server worth pointing at. Someone pasting the
    dashboard's address bar (`http://localhost:3001`) would otherwise get a
    404 on `/chat/completions` with nothing to explain it, so a base whose last
    path segment is not a `vN` version gets one appended.
    """
    u = (url or "").strip().rstrip("/")
    if not u:
        return u
    last = u.split("?")[0].rstrip("/").rsplit("/", 1)[-1].lower()
    if re.fullmatch(r"v\d+", last):
        return u
    return f"{u}/v1"


def get_llm_settings() -> tuple[str, str]:
    """Returns (base_url, model_name) for the selected provider.

    Precedence, highest first: provider-specific env var, generic env var,
    config file, built-in default. The URL is normalised for the two
    OpenAI-compatible providers only — Ollama's `/api/*` surface has no `/v1`
    and appending one would break it.
    """
    provider = get_llm_provider()
    cfg      = _load_config()
    defaults = _DEFAULTS[provider]
    penv     = _PROVIDER_ENV.get(provider, {})

    url = (_env(penv.get("url", "")) or _env(_ENV_URL)
           or str(cfg.get("llm_url", "")).strip() or defaults["url"])
    model = (_env(penv.get("model", "")) or _env(_ENV_MODEL)
             or str(cfg.get("llm_model", "")).strip() or defaults["model"])

    if provider in (OPENAI, FREELLMAPI):
        url = _normalize_base(url)
    else:
        url = url.rstrip("/")
    return url, model


def get_api_key() -> str:
    """The bearer token for the selected provider ('' when none is set).

    Ollama on localhost needs none. An OpenAI-compatible box may or may not.
    FreeLLMAPI requires the unified key from its Keys page. Never printed.
    """
    provider = get_llm_provider()
    cfg      = _load_config()
    penv     = _PROVIDER_ENV.get(provider, {})

    key = _env(penv.get("api_key", "")) or _env(_ENV_API_KEY)
    if key:
        return key

    # Provider-specific config key first, then the shared one.
    key = cfg.get(f"{provider}_api_key") or cfg.get("llm_api_key") or ""
    return str(key).strip()


def _auth_headers() -> dict:
    key = get_api_key()
    return {"Authorization": f"Bearer {key}"} if key else {}


def get_max_tokens(kind: str = "chat") -> int:
    """Token ceiling for a reply. `kind` is 'chat' or 'text'."""
    default = _MAX_TOKENS_TEXT if kind == "text" else _MAX_TOKENS_CHAT
    try:
        val = int(_env(_ENV_MAX_TOKENS) or _load_config().get("llm_max_tokens", 0))
    except (TypeError, ValueError):
        return default
    return val if val > 0 else default


def is_configured() -> bool:
    """True when the selected provider has everything it needs to be asked.

    Ollama needs nothing but a reachable server; the two hosted/remote
    providers need a key. Used to decide whether the free rung on the Gemini
    ladder is even worth offering.
    """
    provider = get_llm_provider()
    if provider == OLLAMA:
        return True
    return bool(get_api_key())


def provider_label() -> str:
    """One-line description for logs and diagnostics. Never contains the key."""
    url, model = get_llm_settings()
    provider   = get_llm_provider()
    auth       = "bearer key" if get_api_key() else "no key"
    return f"{provider} @ {url} model={model} ({auth})"


# ── Health ────────────────────────────────────────────────────────────────────

def _openai_models(url: str, key: str, timeout: int = 5) -> list[str]:
    """Model ids from GET {base}/models. Empty list on any failure.

    FreeLLMAPI lists its whole catalog here, including entries that cannot
    serve a request yet; the router's virtual entries (`auto`, `fusion`,
    `auto:<profile>`) are always present, which is why `auto` validates.
    """
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        resp = requests.get(f"{url}/models", headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []
    rows = data.get("data") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows:
        mid = row.get("id") if isinstance(row, dict) else None
        if mid:
            out.append(str(mid))
    return out


def list_models(timeout: int = 5) -> list[str]:
    """Every model the configured provider reports. Ollama: pulled models."""
    provider   = get_llm_provider()
    url, _     = get_llm_settings()
    if provider in (OPENAI, FREELLMAPI):
        return _openai_models(url, get_api_key(), timeout=timeout)
    try:
        resp = requests.get(f"{url}/api/tags", timeout=timeout)
        resp.raise_for_status()
        return [m.get("name", "") for m in resp.json().get("models", [])]
    except Exception:
        return []


def ensure_ollama_running(timeout: int = 15) -> bool:
    """
    For Ollama: ping /api/tags; auto-launch 'ollama serve' if not running.
    For OpenAI-compatible providers: just ping /v1/models (server must be started manually).
    Returns True if the LLM server is reachable.
    """
    url, _   = get_llm_settings()
    provider = get_llm_provider()

    if provider in (OPENAI, FREELLMAPI):
        # OpenAI-compatible servers (LM Studio, LocalAI, the FreeLLMAPI router)
        # must be started by the user — we just check if they're reachable.
        health  = f"{url}/models"
        headers = _auth_headers()
        try:
            ok = requests.get(health, headers=headers, timeout=5).status_code == 200
            if ok:
                print(f"[LLM] {provider} server reachable at {url}")
            else:
                print(f"[LLM] Server at {url} returned non-200.  Is it running?")
            return ok
        except Exception:
            if provider == FREELLMAPI:
                print(
                    f"[LLM] Cannot reach FreeLLMAPI at {url}.\n"
                    "      Start the router (https://github.com/tashfeenahmed/freellmapi),\n"
                    "      add provider keys on its Keys page, and copy the unified key\n"
                    "      into config/api_keys.json as \"llm_api_key\"."
                )
            else:
                print(
                    f"[LLM] Cannot reach OpenAI-compatible server at {url}.\n"
                    "      Make sure LM Studio / LocalAI / Jan is running and the server is started."
                )
            return False

    # ── Ollama ──────────────────────────────────────────────────────────────
    health = f"{url}/api/tags"

    def _is_up() -> bool:
        try:
            return requests.get(health, timeout=3).status_code == 200
        except Exception:
            return False

    if _is_up():
        return True

    print("[LLM] Ollama not running — launching 'ollama serve'…")
    try:
        kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        subprocess.Popen(["ollama", "serve"], **kwargs)
    except FileNotFoundError:
        print("[LLM] 'ollama' command not found. Install Ollama from https://ollama.com")
        return False
    except Exception as e:
        print(f"[LLM] Could not launch Ollama: {e}")
        return False

    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(1.0)
        if _is_up():
            print("[LLM] Ollama started successfully.")
            return True

    print("[LLM] Ollama did not respond within the timeout.")
    return False


# Name the function is really doing the job of, for callers that are not
# Ollama-specific. Same behaviour, honest name.
ensure_server_running = ensure_ollama_running


def warmup_model(system_prompt: str | None = None) -> bool:
    """
    Pre-load the model AND prime Ollama's KV prefix cache.

    Why the system_prompt matters
    ─────────────────────────────
    Ollama caches the KV attention state of the prompt prefix across requests.
    If warmup includes the same system prompt that real requests will use, Ollama
    evaluates those tokens ONCE at startup.  Every subsequent request only needs
    to evaluate the small delta (user message ± time context) instead of the full
    300-500 token system prompt → drops first-token latency from ~17 s to <1 s.

    Pass the *static* part of the system prompt (the JARVIS protocol text, without
    timestamps or per-minute context) so the prefix stays valid across calls.
    """
    url, model = get_llm_settings()
    provider   = get_llm_provider()
    print(f"[LLM] Warming up '{model}' ({provider})…")

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": "hi"})

    if provider in (OPENAI, FREELLMAPI):
        # OpenAI-compatible: just fire a minimal request to ensure the model is
        # reachable and the routing chain has a healthy key. The router manages
        # its own cache; there is nothing here to prime.
        payload = {
            "model":      model,
            "messages":   messages,
            "stream":     False,
            "max_tokens": 1,
        }
        try:
            resp = _openai_post(url, payload, timeout=180, label="warmup")
            _log_routed_via(resp, provider)
            print(f"[LLM] '{model}' ready ({provider} server).")
            return True
        except Exception as e:
            print(f"[LLM] Warmup failed (non-fatal): {e}")
            return False

    # ── Ollama ──────────────────────────────────────────────────────────────
    payload = {
        "model":      model,
        "messages":   messages,
        "stream":     False,
        "keep_alive": -1,
        # num_gpu:99 → push ALL transformer layers to GPU (Ollama caps at available)
        # This is safe even without a GPU — Ollama silently ignores if n_gpu_layers=0
        "options":    {"num_predict": 1, "num_gpu": 99},
    }
    try:
        resp = requests.post(f"{url}/api/chat", json=payload, timeout=180)
        resp.raise_for_status()
        print(f"[LLM] '{model}' loaded and KV cache primed.")
        return True
    except Exception as e:
        print(f"[LLM] Warmup failed (non-fatal): {e}")
        return False


def check_model_available(log: Callable | None = None) -> bool:
    """
    Returns True if the configured model is already pulled in Ollama.
    Logs an actionable warning (to console + optional UI callback) if not.

    For FreeLLMAPI the same question is answerable, and worth asking: GET
    /v1/models lists the catalog the router can actually reach, so a model id
    that is not in it (or a typo'd routing profile) is caught here instead of
    as a 400 on the first real request. `auto`, `auto:*` and `fusion` are
    router entries rather than catalog rows and are always accepted.
    """
    provider   = get_llm_provider()
    url, model = get_llm_settings()

    if provider == FREELLMAPI:
        if model == "fusion" or model == "auto" or model.lower().startswith("auto:"):
            return True
        models = list_models()
        if not models:            # unreachable, or the list call failed
            return True           # non-blocking: the request will say more
        if model in models:
            return True
        warn = (f"WRN: Model '{model}' is not in the FreeLLMAPI catalog.\n"
                f"     Available (first 8): {', '.join(models[:8])}\n"
                "     Fix: set \"llm_model\": \"auto\" and let the router choose.")
        print(warn)
        if log:
            log(f"WRN: '{model}' not in the FreeLLMAPI catalog — try \"auto\"")
        return False

    if provider != OLLAMA:
        return True

    try:
        resp = requests.get(f"{url}/api/tags", timeout=5)
        resp.raise_for_status()
        pulled = [m.get("name", "") for m in resp.json().get("models", [])]
        model_base = model.split(":")[0]
        found = any(
            m == model or m == model_base or m.startswith(model_base + ":")
            for m in pulled
        )
        if not found:
            available = ", ".join(pulled) if pulled else "none"
            warn = (
                f"WRN: Model '{model}' is not pulled in Ollama.\n"
                f"     Available: {available}\n"
                f"     Fix: ollama pull {model}"
            )
            print(warn)
            if log:
                log(f"WRN: '{model}' not found — run: ollama pull {model}")
        return found
    except Exception:
        return True   # Ollama might still be starting up; non-blocking


# ── OpenAI-compatible transport ───────────────────────────────────────────────
# Shared by the "openai" and "freellmapi" providers. The wire format is
# identical; what differs is the default base URL, the model name, and the
# wording of the errors, which is handled at the call site.

def _log_routed_via(resp, provider: str) -> None:
    """Say which provider actually served the call.

    FreeLLMAPI sets `X-Routed-Via: <platform>/<model>` on every response, and
    `X-Fallback-Attempts` when the first choice was rate-limited and the router
    moved on. That is the only way to tell from the outside whether a slow
    answer was one provider stalling or three failing in turn.
    """
    if provider != FREELLMAPI:
        return
    try:
        via = resp.headers.get("x-routed-via") or resp.headers.get("X-Routed-Via")
        if via:
            hops = resp.headers.get("x-fallback-attempts")
            tail = " (+%s fallback hop(s))" % hops if hops else ""
            print(f"[LLM] FreeLLMAPI served by {via}{tail}")
    except Exception:
        pass


def _error_detail(resp) -> str:
    """The server's own message if it sent one in the OpenAI error envelope."""
    try:
        data = resp.json()
    except Exception:
        return (resp.text or "")[:200]
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        msg = err.get("message") or ""
        typ = err.get("type") or ""
        return f"{msg} [{typ}]" if typ else str(msg)
    if isinstance(err, str):
        return err
    return (resp.text or "")[:200]


def _retry_after(resp) -> float:
    """Seconds from a Retry-After header, clamped. 0 when there is none."""
    raw = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
    if not raw:
        return 0.0
    try:
        return max(0.0, min(_RETRY_MAX_WAIT, float(str(raw).strip())))
    except (TypeError, ValueError):
        return 0.0


def _describe_http_error(provider: str, resp, what: str) -> str:
    """Turn a status code into the sentence that tells the user what to do."""
    code   = resp.status_code
    detail = _error_detail(resp)
    if code == 401:
        where = ("config/api_keys.json → \"llm_api_key\" (or FREELLMAPI_API_KEY)"
                 if provider == FREELLMAPI else
                 "config/api_keys.json → \"llm_api_key\" (or OPENAI_API_KEY)")
        return (f"{provider} rejected the API key (HTTP 401: {detail}). "
                f"Set it in {where}.")
    if code == 404:
        return (f"{what}: {provider} has no such endpoint or model "
                f"(HTTP 404: {detail}). Check \"llm_url\" and \"llm_model\"; "
                f"GET /v1/models lists what is available.")
    if code == 429:
        wait = _retry_after(resp)
        extra = f" The server asked to wait {wait:.0f}s." if wait else ""
        return (f"{provider} is rate-limited (HTTP 429: {detail}).{extra} "
                "Every free tier in the pool is momentarily exhausted — "
                "try again shortly.")
    if code == 400:
        return (f"{provider} rejected the request (HTTP 400: {detail}). "
                "For FreeLLMAPI this usually means an unknown routing "
                "profile in \"llm_model\" (e.g. \"auto:coding\" that does "
                "not exist on the dashboard).")
    return f"{provider} HTTP error: {code} — {detail}"


def _openai_post(url: str, payload: dict, timeout: int, label: str = "request"):
    """POST {base}/chat/completions with auth, and one rate-limit retry.

    Returns the requests.Response on success; raises RuntimeError with an
    actionable message on anything else. Streaming callers pass stream=True in
    the payload and use `stream=True` on the request themselves.
    """
    endpoint = f"{url}/chat/completions"
    headers  = _auth_headers()
    provider = get_llm_provider()

    last: Exception | None = None
    for attempt in range(_RETRY_LIMIT_ATTEMPTS + 1):
        try:
            resp = requests.post(endpoint, json=payload, headers=headers,
                                 timeout=timeout)
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(_connection_message(provider, url, e))
        except requests.exceptions.Timeout:
            raise RuntimeError(f"{provider} {label} timed out after {timeout}s.")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"{provider} {label} failed: {e}")

        if resp.status_code == 429 and attempt < _RETRY_LIMIT_ATTEMPTS:
            wait = _retry_after(resp) or _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
            print(f"[LLM] {provider} 429 — retrying in {wait:.1f}s "
                  f"(attempt {attempt + 1}/{_RETRY_LIMIT_ATTEMPTS})")
            time.sleep(wait)
            last = RuntimeError(_describe_http_error(provider, resp, label))
            continue

        if resp.status_code >= 400:
            raise RuntimeError(_describe_http_error(provider, resp, label))
        return resp

    raise last or RuntimeError(f"{provider} {label} failed after retries.")


def _connection_message(provider: str, url: str, exc: Exception | None = None) -> str:
    if provider == FREELLMAPI:
        return (
            f"Cannot reach FreeLLMAPI at {url}. Start the router, add at least "
            "one free provider key on its Keys page, and copy the unified key "
            f"into config/api_keys.json as \"llm_api_key\". ({exc})"
            if exc else
            f"Cannot reach FreeLLMAPI at {url}. Start the router and add a "
            "provider key on its Keys page."
        )
    return (
        f"Cannot reach OpenAI-compatible server at {url}. Make sure "
        f"LM Studio / LocalAI / Jan is running and the server is started. ({exc})"
        if exc else
        f"Cannot reach OpenAI-compatible server at {url}. Make sure "
        "LM Studio / LocalAI / Jan is running and the server is started."
    )


# ── Tool calls ────────────────────────────────────────────────────────────────

def _parse_tool_arguments(raw):
    """OpenAI sends tool arguments as a JSON *string*; Ollama sends a dict.

    Normalise to a dict so every caller downstream sees one shape. A model that
    emits broken JSON (a trailing comma, an unescaped quote, a half-streamed
    object) must not take the tool down: try the string as-is, then try the
    outermost {...} slice, and if neither parses hand the raw string back —
    which is what main.py's `_execute_tool` already copes with.
    """
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    if not isinstance(raw, str):
        return {}
    text = raw.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return text


def _normalise_openai_tool_calls(raw_tc) -> list:
    """OpenAI tool_calls → the Ollama-shaped dict the rest of Mark-LV expects."""
    out = []
    for t in raw_tc or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") or {}
        name = fn.get("name") or ""
        if not name:
            continue                      # a fragment with no name is not a call
        out.append({
            "id":       t.get("id", "") or "",
            "function": {"name": name,
                         "arguments": _parse_tool_arguments(fn.get("arguments"))},
        })
    return out


# ── Non-streaming ─────────────────────────────────────────────────────────────

def call_llm(
    messages: list,
    tools:    list | None = None,
    timeout:  int = 120,
) -> dict:
    """
    Non-streaming chat request.  Routes to Ollama, an OpenAI-compatible server,
    or FreeLLMAPI.

    Returns:
        {"content": str, "tool_calls": list}
    """
    url, model = get_llm_settings()
    provider   = get_llm_provider()

    if provider in (OPENAI, FREELLMAPI):
        payload: dict = {
            "model":      model,
            "messages":   messages,
            "stream":     False,
            "max_tokens": get_max_tokens("chat"),
        }
        if tools:
            payload["tools"]       = tools
            payload["tool_choice"] = "auto"
        try:
            resp = _openai_post(url, payload, timeout=timeout)
            _log_routed_via(resp, provider)
            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            msg    = choice.get("message") or {}
            return {
                "content":    (msg.get("content") or "").strip(),
                "tool_calls": _normalise_openai_tool_calls(msg.get("tool_calls")),
            }
        except RuntimeError:
            raise
        except ValueError as e:                      # body was not JSON
            raise RuntimeError(f"{provider} returned a non-JSON reply: {e}")
        except Exception as e:
            raise RuntimeError(f"{provider} LLM call failed: {e}")

    # ── Ollama ──────────────────────────────────────────────────────────────
    endpoint = f"{url}/api/chat"
    payload = {
        "model":      model,
        "messages":   messages,
        "stream":     False,
        "keep_alive": -1,
        "options":    {"num_predict": get_max_tokens("chat"), "num_gpu": 99},
    }
    if tools:
        payload["tools"] = tools

    try:
        resp = requests.post(endpoint, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        msg  = data.get("message", {})
        return {
            "content":    (msg.get("content") or "").strip(),
            "tool_calls": msg.get("tool_calls") or [],
        }
    except requests.exceptions.ConnectionError as e:
        print(f"[LLM] ConnectionError — trying to restart Ollama… ({e})")
        if ensure_ollama_running():
            try:
                resp = requests.post(endpoint, json=payload, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                msg  = data.get("message", {})
                return {
                    "content":    (msg.get("content") or "").strip(),
                    "tool_calls": msg.get("tool_calls") or [],
                }
            except Exception:
                pass
        raise RuntimeError(
            f"Cannot connect to Ollama at {url}. "
            "Make sure Ollama is installed and run: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama request timed out after 120 s.")
    except requests.exceptions.HTTPError as e:
        print(f"[LLM] HTTPError: {e.response.status_code} — {e.response.text[:200]}")
        raise RuntimeError(f"Ollama HTTP error: {e.response.status_code}")
    except Exception as e:
        print(f"[LLM] Unexpected error: {type(e).__name__}: {e}")
        raise RuntimeError(f"LLM call failed: {e}")


def call_llm_text(
    prompt:  str,
    system:  str | None = None,
    model:   str | None = None,
    timeout: int = 120,
) -> str:
    """
    Simple text-only generation (no tools).
    Used by the task engine, the research brief, code_helper and dev_agent.

    This used to build an Ollama `/api/chat` payload unconditionally, so it
    silently broke the moment "llm_provider" was set to anything else. It now
    routes exactly like call_llm() and returns the reply text.
    """
    url, default_model = get_llm_settings()
    provider           = get_llm_provider()
    m                  = model or default_model

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    if provider in (OPENAI, FREELLMAPI):
        payload = {
            "model":      m,
            "messages":   messages,
            "stream":     False,
            "max_tokens": get_max_tokens("text"),
        }
        try:
            resp = _openai_post(url, payload, timeout=timeout, label="text call")
            _log_routed_via(resp, provider)
            data   = resp.json()
            choice = (data.get("choices") or [{}])[0]
            return ((choice.get("message") or {}).get("content") or "").strip()
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"{provider} text call failed: {e}")

    # ── Ollama ──────────────────────────────────────────────────────────────
    endpoint = f"{url}/api/chat"
    payload = {"model": m, "messages": messages, "stream": False, "keep_alive": -1,
               "options": {"num_predict": get_max_tokens("text")}}

    try:
        resp = requests.post(endpoint, json=payload, timeout=timeout)
        resp.raise_for_status()
        return (resp.json().get("message", {}).get("content") or "").strip()
    except requests.exceptions.ConnectionError:
        if ensure_ollama_running():
            try:
                resp = requests.post(endpoint, json=payload, timeout=timeout)
                resp.raise_for_status()
                return (resp.json().get("message", {}).get("content") or "").strip()
            except Exception:
                pass
        raise RuntimeError(
            f"Cannot connect to Ollama at {url}. "
            "Make sure Ollama is installed and run: ollama serve"
        )
    except Exception as e:
        raise RuntimeError(f"LLM text call failed: {e}")


# ── Streaming ─────────────────────────────────────────────────────────────────

def _stream_openai_compat(
    messages: list,
    tools:    list | None,
    timeout:  int,
) -> Generator[dict, None, None]:
    """
    Streaming backend for any OpenAI-compatible server (LM Studio, LocalAI,
    Jan, the FreeLLMAPI router…).

    Parses Server-Sent Events (SSE) and accumulates streaming tool-call
    fragments so the output format is identical to the Ollama backend:
    complete sentences as they arrive, then one "done" event.
    """
    url, model = get_llm_settings()
    provider   = get_llm_provider()
    endpoint   = f"{url}/chat/completions"
    headers    = _auth_headers()

    payload: dict = {
        "model":      model,
        "messages":   messages,
        "stream":     True,
        "max_tokens": get_max_tokens("chat"),
    }
    if tools:
        payload["tools"]       = tools
        payload["tool_choice"] = "auto"

    try:
        with requests.post(endpoint, json=payload, headers=headers,
                           timeout=timeout, stream=True) as resp:
            if resp.status_code >= 400:
                raise RuntimeError(_describe_http_error(provider, resp, "stream"))
            _log_routed_via(resp, provider)
            full_content = ""
            buf          = ""
            # tool_call fragments: index → {"id", "function": {"name", "arguments"}}
            tc_fragments: dict[int, dict] = {}

            for raw in resp.iter_lines():
                if not raw:
                    continue
                # SSE lines look like: b"data: {...}" or b"data: [DONE]"
                line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choice = (chunk.get("choices") or [{}])[0]
                delta  = choice.get("delta", {}) or {}
                text   = delta.get("content") or ""

                full_content += text
                buf          += text

                # Accumulate sentence boundaries for streaming TTS
                while True:
                    m = _SENT_END.search(buf)
                    if not m:
                        break
                    sentence = buf[: m.start() + 1].strip()
                    buf      = buf[m.end():]
                    if sentence:
                        yield {"type": "sentence", "text": sentence}

                # Accumulate streaming tool-call fragments
                for tc in (delta.get("tool_calls") or []):
                    idx = tc.get("index", 0)
                    if idx not in tc_fragments:
                        tc_fragments[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
                    frag = tc_fragments[idx]
                    frag["id"] = frag["id"] or tc.get("id", "")
                    fn = tc.get("function", {}) or {}
                    frag["function"]["name"]      += fn.get("name") or ""
                    frag["function"]["arguments"] += fn.get("arguments") or ""

                finish = choice.get("finish_reason")
                if finish in ("stop", "tool_calls", "length"):
                    break

            # Flush any trailing content
            if buf.strip():
                yield {"type": "sentence", "text": buf.strip()}

            tool_calls = []
            for idx in sorted(tc_fragments):
                frag = tc_fragments[idx]
                name = frag["function"]["name"]
                if not name:
                    continue
                tool_calls.append({
                    "id":       frag["id"],
                    "function": {"name": name,
                                 "arguments": _parse_tool_arguments(frag["function"]["arguments"])},
                })

            yield {
                "type":       "done",
                "content":    full_content.strip(),
                "tool_calls": tool_calls,
            }

    except requests.exceptions.ConnectionError:
        raise RuntimeError(_connection_message(provider, url))
    except requests.exceptions.Timeout:
        raise RuntimeError(f"{provider} stream timed out after {timeout}s.")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"{provider} HTTP error: {e.response.status_code}")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"{provider} stream failed: {e}")


# Backwards-compatible name: this was the OpenAI-only streamer.
_stream_openai = _stream_openai_compat


def call_llm_stream(
    messages: list,
    tools:    list | None = None,
    timeout:  int = 120,
) -> Generator[dict, None, None]:
    """
    Streaming chat request.  Routes to Ollama, an OpenAI-compatible server, or
    FreeLLMAPI.

    Yields:
        {"type": "sentence", "text": str}   — each complete sentence as it arrives
        {"type": "done", "content": str, "tool_calls": list}  — when stream ends

    Sentences are split on [.!?] + whitespace so TTS can start immediately.
    Tool calls always appear in the final "done" event.
    """
    provider = get_llm_provider()
    if provider in (OPENAI, FREELLMAPI):
        yield from _stream_openai_compat(messages, tools, timeout)
        return

    url, model = get_llm_settings()
    endpoint   = f"{url}/api/chat"

    payload: dict = {
        "model":      model,
        "messages":   messages,
        "stream":     True,
        "keep_alive": -1,
        # 150 tokens ≈ 100 words ≈ 3-4 sentences — enough for any voice reply.
        # num_gpu:99 pushes all layers to GPU; num_thread removed (Ollama auto-tunes).
        "options":    {"num_predict": get_max_tokens("chat"), "num_gpu": 99},
    }
    if tools:
        payload["tools"] = tools

    def _do_stream() -> Generator[dict, None, None]:
        with requests.post(endpoint, json=payload, timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            full_content = ""
            tool_calls:  list = []
            buf          = ""

            for raw in resp.iter_lines():
                if not raw:
                    continue
                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg   = chunk.get("message", {})
                delta = msg.get("content") or ""

                full_content += delta
                buf          += delta

                # Yield complete sentences as they accumulate
                while True:
                    m = _SENT_END.search(buf)
                    if not m:
                        break
                    sentence = buf[: m.start() + 1].strip()
                    buf      = buf[m.end() :]
                    if sentence:
                        yield {"type": "sentence", "text": sentence}

                tc = msg.get("tool_calls")
                if tc:
                    tool_calls.extend(tc)

                if chunk.get("done"):
                    if buf.strip():
                        yield {"type": "sentence", "text": buf.strip()}

                    yield {
                        "type":       "done",
                        "content":    full_content.strip(),
                        "tool_calls": tool_calls,
                    }
                    return

    try:
        yield from _do_stream()
    except requests.exceptions.ConnectionError as e:
        print(f"[LLM] Stream ConnectionError — trying to restart Ollama… ({e})")
        if ensure_ollama_running():
            yield from _do_stream()
            return
        raise RuntimeError(
            f"Cannot connect to Ollama at {url}. "
            "Make sure Ollama is installed and run: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama stream timed out.")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Ollama HTTP error: {e.response.status_code}")
    except Exception as e:
        print(f"[LLM] Stream error: {type(e).__name__}: {e}")
        raise RuntimeError(f"LLM stream failed: {e}")


# ── Diagnostics ───────────────────────────────────────────────────────────────

def diagnose(timeout: int = 5) -> dict:
    """Everything worth knowing about the configured provider, key redacted.

    Powers `python -m core.llm_client` and the docs' troubleshooting table:
    one call that says which provider, which URL, which model, whether a key
    is present, whether the server answers, and what it reports.
    """
    provider   = get_llm_provider()
    url, model = get_llm_settings()
    out = {
        "provider":     provider,
        "base_url":     url,
        "model":        model,
        "api_key_set":  bool(get_api_key()),
        "max_tokens":   {"chat": get_max_tokens("chat"),
                         "text": get_max_tokens("text")},
        "reachable":    False,
        "models":       [],
        "error":        "",
    }
    try:
        if provider in (OPENAI, FREELLMAPI):
            headers = _auth_headers()
            resp = requests.get(f"{url}/models", headers=headers, timeout=timeout)
            out["reachable"] = resp.status_code == 200
            if not out["reachable"]:
                out["error"] = f"GET /models → HTTP {resp.status_code}"
        else:
            resp = requests.get(f"{url}/api/tags", timeout=timeout)
            out["reachable"] = resp.status_code == 200
            if not out["reachable"]:
                out["error"] = f"GET /api/tags → HTTP {resp.status_code}"
        if out["reachable"]:
            out["models"] = list_models(timeout=timeout)[:20]
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


if __name__ == "__main__":
    # `python -m core.llm_client` — print the configuration and reach the
    # server once. Deliberately makes no generation call: a diagnostic that
    # spends quota is a diagnostic nobody runs twice.
    print(json.dumps(diagnose(), indent=2))
