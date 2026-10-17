"""The research action: bounds, reuse of the existing search, and where the
answer goes.

The behaviour that matters is that this is not a second search stack — it asks
the SAME web_search action the rest of the app uses, and it files the brief in
the SAME memory store recall_memory already reads. A research tool that quietly
grew its own versions of both would drift from them within a release.
"""
import pytest

from actions import research
from memory import memory_manager


class FakePlayer:
    def __init__(self):
        self.logs, self.contents = [], []

    def write_log(self, text):
        self.logs.append(text)

    def show_content(self, label, text):
        self.contents.append((label, text))


@pytest.fixture
def no_model(monkeypatch):
    """Decomposition and synthesis off the network, with a predictable reply."""
    from core import gemini

    def fake_text(contents, tier=gemini.FAST, config=None, timeout_ms=0, **kw):
        text = contents if isinstance(contents, str) else str(contents)
        if "search queries" in text:
            # More angles than any depth asks for, so it is the depth cap that
            # decides how many are used.
            return ("1. heat pump cost\n2. heat pump efficiency\n"
                    "3. heat pump install\n4. heat pump noise\n"
                    "5. heat pump grants\n6. heat pump lifespan")
        return "Heat pumps cost around eight thousand installed."

    monkeypatch.setattr(gemini, "text", fake_text)


@pytest.fixture
def fake_search(monkeypatch):
    """Replace the search backend and record what was asked of it."""
    from actions import web_search

    asked = []

    def fake(query):
        asked.append(query)
        return f"Findings about: {query}"

    monkeypatch.setattr(web_search, "_research", fake)
    return asked


@pytest.fixture
def fake_memory(monkeypatch):
    stored = {}
    monkeypatch.setattr(memory_manager, "remember",
                        lambda key, value, category="notes":
                        stored.update({key: (value, category)}) or "ok")
    return stored


# ── It asks the existing search action ───────────────────────────────────────

def test_sub_questions_go_through_web_search(no_model, fake_search, fake_memory):
    out = research.research({"topic": "heat pumps for this house"})

    assert fake_search == ["heat pump cost", "heat pump efficiency",
                           "heat pump install"]      # 'standard' = three angles
    assert "eight thousand" in out


def test_depth_sets_the_number_of_angles(no_model, fake_memory, monkeypatch):
    from actions import web_search
    asked = []
    monkeypatch.setattr(web_search, "_research",
                        lambda q: asked.append(q) or f"about {q}")

    research.research({"topic": "solar", "depth": "quick"})
    assert len(asked) == 2

    asked.clear()
    research.research({"topic": "solar", "depth": "deep"})
    assert len(asked) == 4


def test_an_unknown_depth_falls_back_to_standard(no_model, fake_search, fake_memory):
    research.research({"topic": "heat pumps", "depth": "extreme"})
    assert len(fake_search) == research._DEPTHS["standard"]["questions"]


# ── It is bounded ────────────────────────────────────────────────────────────

def test_a_hanging_search_does_not_hang_the_action(no_model, fake_memory, monkeypatch):
    from actions import web_search

    def hang(query):
        import time
        time.sleep(30)
        return "never arrives"

    monkeypatch.setattr(web_search, "_research", hang)
    monkeypatch.setattr(research, "_DEPTHS",
                        {"quick": {"questions": 2, "per_query": 0.2, "budget": 1.0},
                         "standard": {"questions": 2, "per_query": 0.2, "budget": 1.0},
                         "deep": {"questions": 2, "per_query": 0.2, "budget": 1.0}})

    out = research.research({"topic": "anything"})

    assert "could not find anything" in out.lower()


def test_a_failing_search_is_skipped_not_fatal(no_model, fake_memory, monkeypatch):
    from actions import web_search

    def flaky(query):
        if "cost" in query:
            raise RuntimeError("grounding quota exhausted")
        return f"about {query}"

    monkeypatch.setattr(web_search, "_research", flaky)

    out = research.research({"topic": "heat pumps"})

    assert out                       # still produced a brief
    assert "could not find" not in out.lower()


def test_empty_search_results_are_not_treated_as_findings(no_model, fake_memory,
                                                          monkeypatch):
    from actions import web_search
    monkeypatch.setattr(web_search, "_research", lambda q: "No results found.")

    out = research.research({"topic": "nothing at all"})

    assert "could not find anything" in out.lower()


def test_decomposition_failure_still_searches_the_topic(fake_memory, monkeypatch):
    """A model that will not produce a clean list must not turn a research
    request into an apology — one search on the topic as asked is an answer."""
    from actions import web_search
    from core import gemini

    asked = []

    def boom(*a, **kw):
        raise RuntimeError("no model available")

    monkeypatch.setattr(gemini, "text", boom)
    monkeypatch.setattr(web_search, "_research",
                        lambda q: asked.append(q) or f"about {q}")

    out = research.research({"topic": "heat pumps", "focus": "cost"})

    assert asked == ["heat pumps cost"]
    assert out


# ── Where the answer goes ────────────────────────────────────────────────────

def test_the_full_brief_reaches_the_content_panel(no_model, fake_search):
    player = FakePlayer()

    research.research({"topic": "heat pumps"}, player=player)

    assert len(player.contents) == 1
    label, text = player.contents[0]
    assert label.startswith("RESEARCH —")
    assert "heat pump cost" in text


def test_the_brief_is_stored_in_the_existing_memory_store(no_model, fake_search,
                                                          fake_memory):
    research.research({"topic": "Heat Pumps For The House"})

    key, (value, category) = next(iter(fake_memory.items()))
    assert key == "research_heat_pumps_for_the_house"
    assert category == "notes"
    assert "eight thousand" in value


def test_nothing_is_stored_when_there_is_nothing_to_store(no_model, fake_memory,
                                                          monkeypatch):
    from actions import web_search
    monkeypatch.setattr(web_search, "_research", lambda q: "No results found.")

    research.research({"topic": "nothing"})

    assert fake_memory == {}


def test_the_run_is_logged(no_model, fake_search, fake_memory):
    player = FakePlayer()

    research.research({"topic": "heat pumps", "focus": "cost"}, player=player)

    assert any("heat pumps" in line for line in player.logs)


def test_no_topic_asks_for_one():
    assert "Tell me what to look into" in research.research({})


def test_partial_coverage_is_said_out_loud(no_model, fake_memory, monkeypatch):
    """A brief that covered two of four angles must not read as complete."""
    from actions import web_search

    calls = {"n": 0}

    def sometimes(query):
        calls["n"] += 1
        return f"about {query}" if calls["n"] <= 2 else "No results found."

    monkeypatch.setattr(web_search, "_research", sometimes)

    out = research.research({"topic": "heat pumps", "depth": "deep"})

    assert "2 of 4" in out


# ── The declaration ──────────────────────────────────────────────────────────

def test_the_tool_declaration_is_valid_for_the_loader():
    from core.action_loader import _validate
    import types

    record = _validate(types.SimpleNamespace(TOOL=research.TOOL), "research.py")

    assert record.valid, record.error
    assert record.name == "research"
    assert record.parameters["properties"]["topic"]["type"] == "STRING"
    assert record.parameters["required"] == ["topic"]


def test_the_declaration_says_how_it_differs_from_web_search():
    """Both tools are in front of the model at once; without this it picks
    arbitrarily between them."""
    assert "web_search" in research.TOOL["description"]
