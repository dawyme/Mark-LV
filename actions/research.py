"""
actions/research.py — a bounded research brief.

web_search already answers one question. This is for the one that is really
four: "look into heat pumps for this house" needs what they cost, what they
need from the building, which are rated well, and what the running cost looks
like — and an answer that arrives as four disconnected searches is not research.

So: decompose the topic into a handful of sub-questions, run the EXISTING
search action against each (Gemini grounded first, DuckDuckGo behind it —
nothing new is invented here), then synthesise one brief.

THE BOUNDS ARE THE FEATURE
    Sub-questions are capped by depth, each search has its own deadline, and
    the whole thing has a wall clock. A research tool with no ceiling is a way
    to make a voice assistant go quiet for two minutes; the caps are chosen so
    the brief lands inside the pause a person will tolerate, and the brief says
    what it did not have time to cover rather than pretending to be complete.

WHERE THE ANSWER GOES
    The short version is what the assistant says. The full brief goes to the
    content panel on screen and into memory under `notes`, so "what did you
    find about heat pumps?" works next week through the ordinary recall tool —
    no second store, no duplicate memory system.
"""
from __future__ import annotations

import re
import time

# depth → how many sub-questions, and how much of the wall clock each gets.
_DEPTHS = {
    "quick":    {"questions": 2, "per_query": 10.0, "budget": 40.0},
    "standard": {"questions": 3, "per_query": 12.0, "budget": 70.0},
    "deep":     {"questions": 4, "per_query": 14.0, "budget": 100.0},
}
_DEFAULT_DEPTH = "standard"
_BRIEF_WORDS   = 170      # the spoken version — about 45 seconds out loud
_PANEL_CHARS   = 4000     # the on-screen version


def _run_bounded(fn, timeout: float, label: str = "task") -> str:
    """Run `fn` on a worker thread and give up on it after `timeout` seconds.

    A search backend that hangs is worse than one that fails: the assistant
    goes quiet with nothing to say. This is the same shape web_search uses
    internally, kept local so this file does not depend on a private helper.
    """
    import threading

    box: dict = {}

    def runner():
        try:
            box["out"] = fn()
        except BaseException as e:                      # noqa: BLE001
            box["err"] = e

    th = threading.Thread(target=runner, daemon=True, name=f"research-{label}")
    th.start()
    th.join(timeout)
    if th.is_alive():
        return ""
    if "err" in box:
        print(f"[Research] {label}: {type(box['err']).__name__}: {box['err']}")
        return ""
    return str(box.get("out") or "")


# ── Decomposition ─────────────────────────────────────────────────────────────

_DECOMPOSE_SYSTEM = (
    "You split one research topic into distinct sub-questions. Reply with ONLY "
    "a numbered list of short search queries, one per line. No prose, no "
    "commentary, no numbering scheme other than '1. '."
)


def _sub_questions(topic: str, focus: str, count: int) -> list[str]:
    """Ask the model for the sub-questions. Falls back to the topic itself.

    The fallback matters: decomposition is a convenience, and a model that
    refuses to produce a clean list must not turn a research request into an
    apology. One search on the topic as asked is still an answer.
    """
    try:
        from core import gemini
        prompt = (f"Topic: {topic}"
                  + (f"\nFocus on: {focus}" if focus else "")
                  + f"\n\nList {count} distinct search queries that together "
                    f"cover it. One per line.")
        raw = gemini.text(prompt, tier=gemini.FAST,
                          config={"system_instruction": _DECOMPOSE_SYSTEM},
                          timeout_ms=20_000)
    except Exception as e:
        print(f"[Research] decomposition failed: {type(e).__name__}: {e}")
        raw = ""

    out = []
    for line in (raw or "").splitlines():
        item = re.sub(r"^\s*[-*\d.)\s]+", "", line).strip().strip('"')
        if item and item.lower() not in {x.lower() for x in out}:
            out.append(item[:160])
        if len(out) >= count:
            break
    return out or [f"{topic} {focus}".strip()]


# ── Gathering ─────────────────────────────────────────────────────────────────

# ── Synthesis ─────────────────────────────────────────────────────────────────

_SYNTH_SYSTEM = (
    "You write a short research brief for a voice assistant to read aloud. "
    "Plain prose, no headings, no bullet points, no markdown. Lead with the "
    "answer. Give the specific numbers and names the material contains. Say "
    "in one clause what the material did not cover. Never invent a fact, a "
    "figure or a source."
)


def _synthesise(topic: str, focus: str, found: list[tuple[str, str]],
                word_cap: int) -> tuple[str, str]:
    """(spoken brief, full brief). The full one is the material itself when
    synthesis is unavailable, because raw findings beat no findings."""
    material = "\n\n".join(f"### {q}\n{t}" for q, t in found)
    full     = f"Research: {topic}\n{'(' + focus + ')' if focus else ''}\n\n{material}"

    if not found:
        return ("I could not find anything on that just now — the search came "
                "back empty."), ""

    try:
        from core import gemini
        prompt = (
            f"Topic: {topic}"
            + (f"\nFocus: {focus}" if focus else "")
            + f"\n\nWrite a brief of at most {word_cap} words from this "
              f"material. Material:\n\n{material[:12000]}"
        )
        brief = gemini.text(prompt, tier=gemini.SMART,
                            config={"system_instruction": _SYNTH_SYSTEM},
                            timeout_ms=60_000)
    except Exception as e:
        print(f"[Research] synthesis failed: {type(e).__name__}: {e}")
        brief = ""

    brief = (brief or "").strip()
    if not brief:
        # No synthesis: hand back the first findings rather than nothing. The
        # panel still has everything.
        first = found[0][1]
        brief = " ".join(first.split())[:400]
    return brief, full


# ── Persistence ───────────────────────────────────────────────────────────────

def _slug(text: str, limit: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return slug[:limit] or "topic"


def _remember(topic: str, brief: str) -> str:
    """Store the brief through the EXISTING memory store, under `notes`.

    Not a new database and not a new recall tool: memory_manager.remember()
    already writes long_term.json, and recall_memory already searches it, so a
    brief filed here is found next week by the tool the assistant already uses.
    """
    if not brief:
        return ""
    try:
        from memory import memory_manager
        key = f"research_{_slug(topic)}"
        memory_manager.remember(key, brief, category="notes")
        return key
    except Exception as e:
        print(f"[Research] could not store the brief: {e}")
        return ""


# ── Action ────────────────────────────────────────────────────────────────────

def research(parameters: dict, player=None, speak=None,
             response=None, session_memory=None) -> str:
    params = parameters or {}
    topic  = str(params.get("topic", "") or "").strip()
    focus  = str(params.get("focus", "") or "").strip()
    depth  = str(params.get("depth", _DEFAULT_DEPTH) or _DEFAULT_DEPTH).strip().lower()

    if not topic:
        return "Tell me what to look into."

    cfg = _DEPTHS.get(depth, _DEPTHS[_DEFAULT_DEPTH])

    if player:
        try:
            player.write_log(f"[Research] {topic}" + (f" ({focus})" if focus else ""))
        except Exception:
            pass

    started  = time.monotonic()
    queries  = _sub_questions(topic, focus, cfg["questions"])
    found    = []
    for q in queries:
        if time.monotonic() - started > cfg["budget"]:
            print(f"[Research] budget reached after {len(found)}/{len(queries)} "
                  f"sub-questions")
            break
        text = _run_bounded(lambda q=q: _gather_one(q), timeout=cfg["per_query"],
                            label=q[:40])
        if text:
            found.append((q, text))

    brief, full = _synthesise(topic, focus, found, _BRIEF_WORDS)

    if full:
        if player:
            try:
                player.show_content(f"RESEARCH — {topic[:38]}", full[:_PANEL_CHARS])
            except Exception:
                pass
        key = _remember(topic, brief)
        if key:
            print(f"[Research] brief stored as notes/{key}")

    covered = len(found)
    if covered < len(queries) and found:
        brief += (f" (That covers {covered} of {len(queries)} angles — say the "
                  f"word and I will dig into the rest.)")

    print(f"[Research] {topic} → {covered}/{len(queries)} sources, "
          f"{time.monotonic() - started:.1f}s")
    return brief


def _gather_one(query: str) -> str:
    """One sub-question, through the existing search action."""
    try:
        from actions import web_search
    except Exception as e:
        print(f"[Research] web_search unavailable: {e}")
        return ""
    text = (web_search._research(query) or "").strip()
    if not text or text.startswith("No results") or text.startswith("Search failed"):
        return ""
    return text[:1800]


# ── Tool declaration (auto-discovered by core/action_loader.py) ──────────────
TOOL = {
    "name": "research",
    "description": (
        "Investigates a topic properly: splits it into sub-questions, searches "
        "each one, and returns one short brief with the specifics in it — "
        "unlike web_search, which answers a single query. Use it when the user "
        "asks you to look into, investigate, compare options for or get up to "
        "speed on something with more than one side to it. The full brief goes "
        "on screen and is remembered, so it can be recalled later. Depth: "
        "'quick' (2 angles), 'standard' (3, default), 'deep' (4)."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "topic": {
                "type": "STRING",
                "description": "What to investigate",
            },
            "focus": {
                "type": "STRING",
                "description": "Optional angle to weight it towards, e.g. 'cost' or 'for a 1930s house'",
            },
            "depth": {
                "type": "STRING",
                "description": "quick | standard | deep (default standard)",
            },
        },
        "required": ["topic"],
    },
    "handler": research,
}
