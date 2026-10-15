# FreeLLMAPI in Mark-LV

A second language model, for the work that is not the conversation.

Mark-LV's voice and conversation stay exactly where they were: **Gemini Live**,
a native-audio model, with the prebuilt voice you pick in ⚙. Nothing in this
integration speaks, hears, or touches the microphone. FreeLLMAPI is a
*reasoning* provider — it answers the one-shot text calls the app already makes
(planning a task, reading a document, explaining code) when the Gemini text
quota has run out.

---

## 1. What FreeLLMAPI is

[FreeLLMAPI](https://github.com/tashfeenahmed/freellmapi) is an open-source,
**self-hosted** router. You give it free-tier API keys from providers you
already have (Google AI Studio, Groq, Cerebras, Mistral, OpenRouter, Z.ai and
others); it puts them all behind one OpenAI-compatible `/v1` endpoint, picks a
healthy model per request, and fails over to the next provider when one is
rate-limited. It is local-first and single-user — there is no hosted signup
that hands you a key.

### The API contract this integration speaks

Verified against the router's own API reference
(`docs/en/api/01-rest-api.md` in the FreeLLMAPI repository) and its server
source — not assumed:

| | |
|---|---|
| **Base URL** | `http://localhost:3001/v1` (configurable) |
| **Auth** | `Authorization: Bearer <unified key>` from the router's **Keys** page |
| **Chat** | `POST {base}/chat/completions` — OpenAI request and response shape |
| **Models** | `GET {base}/models` — supports `?execution_status=ready` |
| **Model field** | `auto` (router picks), `auto:fast` \| `auto:smart` \| `auto:reliable` \| `auto:balanced` \| `auto:cheap`, `auto:<profile>`, `fusion`, or a catalog model id |
| **Streaming** | supported — SSE, `stream: true`; tool calls arrive as `delta.tool_calls` chunks closed by `finish_reason: "tool_calls"` |
| **Tool calling** | supported — OpenAI `tools` + `tool_choice`, multi-step round-trip |
| **Errors** | OpenAI envelope `{"error": {"message", "type"}}` — `401` authentication, `429` rate limit, `404` unknown model, `400` unknown routing profile |
| **Info headers** | `X-Routed-Via: <platform>/<model>`, plus `X-Fallback-Attempts` / `X-Fallback-Trail` after a failover |

> **About `freellmapi.ai`.** A separate site at `freellmapi.ai` publishes docs
> for a *hosted* variant at `https://api.freellmapi.ai/v1` with `sk_live_…`
> keys and models such as `glm-5.2`. Its docs state that streaming and
> Anthropic-side tool use are **not enabled yet** on that deployment. Because
> the base URL, model and key are all configurable, that variant works through
> this same provider — set `llm_url` to `https://api.freellmapi.ai/v1` and
> `llm_model` to a model it lists. It is **not** the default here: the
> self-hosted router is the documented upstream, and at the time of writing
> `api.freellmapi.ai` was returning Cloudflare 522 (origin down).

---

## 2. Setup

### a. Run the router

```bash
git clone https://github.com/tashfeenahmed/freellmapi.git
cd freellmapi
npm install
cp .env.example .env
echo "ENCRYPTION_KEY=$(node -e "console.log(require('crypto').randomBytes(32).toString('hex'))")" >> .env
npm run dev
```

Dashboard: `http://localhost:3001`. Add at least one free provider key on the
**Keys** page, then copy the **unified API key** shown in that page's header.

### b. Tell Mark-LV about it

`config/api_keys.json` (copy `config/api_keys.example.json` if you have none):

```json
{
    "gemini_api_key": "…your existing Gemini key, unchanged…",
    "llm_provider": "freellmapi",
    "llm_url": "http://localhost:3001/v1",
    "llm_model": "auto",
    "llm_api_key": "freellmapi-…your unified key…"
}
```

Or with environment variables, which take precedence over the file — useful
for keeping the key out of a file entirely:

```bash
export FREELLMAPI_API_KEY="freellmapi-…"
export MARK_LLM_PROVIDER=freellmapi
# optional overrides:
export MARK_LLM_URL="http://localhost:3001/v1"
export MARK_LLM_MODEL="auto"
```

### c. Check it

```bash
python -m core.llm_client        # prints provider, URL, model, reachability
python tests/live_freellmapi_check.py
```

`live_freellmapi_check.py` makes **real** requests: model listing, a
non-streaming completion, the text path, a streamed completion, a tool call,
and a deliberately bad model. It exits non-zero unless every step passed, and
it is the thing to run before claiming the integration works.

---

## 3. Configuration reference

All keys live in `config/api_keys.json`. Every one has a working default, so an
untouched config behaves exactly as it did before.

| Key | Default | Meaning |
|---|---|---|
| `llm_provider` | `"ollama"` | `ollama` \| `openai` \| `freellmapi`. Aliases accepted: `lmstudio`, `localai`, `jan`, `llamacpp`, `vllm` → `openai`; `free-llm-api`, `freellm` → `freellmapi`. Unrecognised → `ollama`. |
| `llm_url` | per provider | Ollama `http://localhost:11434` · OpenAI-compatible `http://localhost:1234` · FreeLLMAPI `http://localhost:3001/v1`. For the two OpenAI-compatible providers a missing `/v1` is appended automatically. |
| `llm_model` | per provider | `llama3.2` · `local-model` · `auto` |
| `llm_api_key` | `""` | Bearer token. Not needed for a local Ollama. `freellmapi_api_key` / `openai_api_key` are honoured first if present. |
| `llm_max_tokens` | chat 150, text 600 | Reply ceiling. |
| `llm_gemini_fallback` | `true` | Set `false` to keep FreeLLMAPI out of the Gemini ladder entirely. |

**Environment variables** (checked before the file): `MARK_LLM_PROVIDER`,
`MARK_LLM_URL`, `MARK_LLM_MODEL`, `MARK_LLM_API_KEY`, `MARK_LLM_MAX_TOKENS`,
plus `FREELLMAPI_API_KEY` / `FREELLMAPI_BASE_URL` / `FREELLMAPI_MODEL` and
`OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`.

**No key is ever printed.** `provider_label()` and `diagnose()` report that a
key is present, never what it is, and error messages name the setting to fix
rather than echoing the credential.

---

## 4. Where the provider is actually used

Two places, and deliberately no others.

### `core/llm_client.py` — the provider

Three backends behind one interface: `call_llm()`, `call_llm_text()`,
`call_llm_stream()`, `list_models()`, `check_model_available()`,
`warmup_model()`, `diagnose()`. Callers do not know which one answered.

**The bug this fixed:** `call_llm_text()` built an Ollama `/api/chat` payload
unconditionally — `keep_alive`, `options.num_predict`, and it read the reply
from `message.content`. Set `llm_provider` to anything else and it posted an
Ollama-shaped body to a server that had never heard of those fields. It now
routes per provider, exactly like `call_llm()`.

### `core/gemini.py` — the last rung of the ladder

`core/gemini.py` is the single funnel every one-shot model call in the app goes
through: `dev_agent`, `code_helper`, `file_processor`, `flight_finder`,
`desktop`, `computer_control`, `web_search`. FreeLLMAPI is appended at the
**bottom** of the `FAST`, `SMART` and `SEARCH` ladders, so all of them inherit
it without being edited — and Gemini always gets the first refusal.

Today, when every Gemini rung is out of quota, `gemini.call()` returns `None`
and the feature behind it simply dies. With the rung configured, it gets an
answer from a different pool instead. Set `"llm_gemini_fallback": false` to
turn that off; setting `llm_provider` back to `ollama` also removes the rung.

### What it is not

- **Not the voice.** Conversation audio stays on Gemini Live
  (`main.py::_build_config`, `speech_config` → `prebuilt_voice_config` →
  `get_voice()`). `tests/test_voice_pipeline_intact.py` fails if that changes.
- **Not in the Live loop.** `main.py` does not import `core/llm_client` at all,
  and `core/llm_client` imports no audio library.
- **Not the personality.** `core/prompt.txt` is untouched.

---

## 5. Failure handling

| Situation | Behaviour |
|---|---|
| `401` | `RuntimeError` naming the setting to fix (`llm_api_key` / `FREELLMAPI_API_KEY`). The key is never echoed. |
| `404` | Names the model or endpoint, and points at `GET /v1/models`. |
| `429` | Retried **once**, honouring `Retry-After` (clamped to 20 s), then a clear rate-limit message. Bounded on purpose — retrying a rate limiter in a loop only extends the cooldown. |
| `400` | Explains the usual cause: an unknown `auto:<profile>` routing profile. |
| Timeout / connection refused | Actionable message: start the router, add a provider key. |
| Non-JSON body | Reported as such, with the first 200 characters. |
| Malformed tool arguments | Parsed if possible, salvaged from the outermost `{…}` if not, otherwise handed back as the raw string — which is what `main.py::_execute_tool` already copes with. Never raises. |
| Unreachable at startup | `ensure_ollama_running()` (also aliased `ensure_server_running()`) pings `GET /v1/models` and says what to do. Non-fatal. |
| Model not in the catalog | `check_model_available()` warns and suggests `auto`; `auto`, `auto:*` and `fusion` are always accepted. |

---

## 6. Security

- The unified key is a **credential**. It lives in `config/api_keys.json`,
  which `.gitignore` already excludes, or in the environment. Never commit it.
- `config/api_keys.example.json` is committed with **placeholders only**.
- Nothing is logged that contains the key.
- The router itself is single-user and guarded only by that key. Keep it on
  `127.0.0.1` unless you have a reason not to, and read FreeLLMAPI's own
  `SECURITY.md` before exposing it.

---

## 7. What was actually tested

Run these yourself; both are in the repository.

```bash
python -m pytest                          # 204 tests, offline, no keys needed
python tests/compat_server_check.py       # 10 checks against a real HTTP server
python tests/live_freellmapi_check.py     # real requests to a real router
```

**`python -m pytest` — 204 passed.** Offline by design; every LLM test drives
the shipped `core/llm_client.py` through a fake transport. Covers provider
selection and aliases, env-over-file precedence, per-provider request bodies
and endpoints, auth headers, `/v1` normalisation, SSE parsing and sentence
splitting, streamed tool-call assembly, tool-argument repair, 401/404/429/
400/timeout/connection failures, retry bounding, the Gemini ladder rung, the
task engine and its confirmation gate, the research action, and a guard over
the voice pipeline.

**`python tests/compat_server_check.py` — 10/10 passed.** A local HTTP server
(stdlib only) implementing the documented surface, with the shipped client
pointed at it over a **real socket**: catalog listing, 401 without the key, 404
for an unknown model, 429 + `Retry-After` then success, non-streaming chat,
streamed sentences, tool calls parsed to a dict, tool-call fragments
reassembled across three SSE chunks, and `call_llm_text()` speaking OpenAI.
The test server rejects `keep_alive` / `options` / `num_predict` with a 400, so
a payload in the wrong dialect cannot pass.

**`python tests/live_freellmapi_check.py` — NOT yet run against a real router.**
This is the honest gap, and it is why that script exists separately. See
*Known limitations* below.

---

## 8. Known limitations

- **No real FreeLLMAPI request has been verified yet.** The sandbox this was
  built in has egress limited to GitHub, npm and PyPI, so it cannot reach
  `localhost:3001` on your machine or any hosted endpoint; `api.freellmapi.ai`
  was returning Cloudflare 522 throughout. The wire contract is verified
  against the documented surface over real sockets, and the request/response
  handling against mocks — but the last step, a real router answering, is
  yours to run: `python tests/live_freellmapi_check.py`.
- **Tool calling depends on which model the router picks.** `auto` may route to
  a model that does not support function calling. If a tool call matters, pin
  a model in `llm_model` that you know supports it.
- **The free rung is text-only.** It cannot carry `grounding_metadata`, so a
  `SEARCH`-tier answer that lands on it has no citations attached. It is still
  better than the `None` it replaces.
- **Images.** `core/gemini.py` converts an inline image into an OpenAI
  `image_url` block, which FreeLLMAPI documents and routes to a vision-capable
  model — but only if you have one enabled in its fallback chain. Without one,
  the router returns a `422 no_vision_model`.
