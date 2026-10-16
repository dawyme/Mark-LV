"""The task engine: planning, bounds, persistence, and the safety line.

A plan is the one place in this app where a model gets to decide what happens
next, more than once, without a human in the loop for each step. These tests
are mostly about the edges of that: what a malformed plan does, how many times
a failing step may be retried, and which steps are allowed to run without
somebody pressing a button.
"""
import json

import pytest

from core import task_engine as engine


@pytest.fixture(autouse=True)
def store(tmp_path):
    engine.set_store_path(tmp_path / "tasks.json")
    yield
    engine.set_store_path(engine._base_dir() / "memory" / "tasks.json")


TOOLS = {"web_search", "file_processor", "file_controller", "computer_settings"}


# ── Parsing a plan ───────────────────────────────────────────────────────────

def test_a_clean_plan_parses():
    steps = engine.parse_plan_json(
        '[{"title": "Find flights", "tool": "web_search", "args": {"query": "LGW to POS"}}]',
        TOOLS)
    assert len(steps) == 1
    assert steps[0].title == "Find flights"
    assert steps[0].tool == "web_search"
    assert steps[0].args == {"query": "LGW to POS"}


def test_a_fenced_plan_parses():
    """Models wrap JSON in markdown whether or not they were asked not to."""
    steps = engine.parse_plan_json(
        '```json\n[{"title": "Read the file", "tool": "file_processor",'
        ' "args": {}}]\n```', TOOLS)
    assert [s.tool for s in steps] == ["file_processor"]


def test_a_plan_wrapped_in_an_object_parses():
    steps = engine.parse_plan_json(
        '{"steps": [{"title": "a", "tool": "web_search"},'
        ' {"title": "b", "tool": "think"}]}', TOOLS)
    assert [s.title for s in steps] == ["a", "b"]


def test_an_unknown_tool_is_dropped_at_plan_time():
    """The failure that matters most: a step naming a tool that does not exist
    looks like progress and then fails three steps in."""
    steps = engine.parse_plan_json(
        '[{"title": "ok", "tool": "web_search"},'
        ' {"title": "nope", "tool": "teleport_user"},'
        ' {"title": "fine", "tool": "think"}]', TOOLS)
    assert [s.title for s in steps] == ["ok", "fine"]


def test_think_is_always_available():
    steps = engine.parse_plan_json('[{"title": "reason", "tool": "think"}]', set())
    assert len(steps) == 1


def test_garbage_produces_no_steps_and_never_raises():
    for raw in ("", None, "I cannot help with that", "[{", "[]", "null",
                '[{"title": "", "tool": "web_search"}]'):
        assert engine.parse_plan_json(raw, TOOLS) == [], repr(raw)


def test_a_plan_is_capped():
    rows = [{"title": f"step {i}", "tool": "web_search"} for i in range(50)]
    steps = engine.parse_plan_json(json.dumps(rows), TOOLS)
    assert len(steps) == engine.MAX_STEPS


def test_non_dict_args_become_an_empty_dict():
    steps = engine.parse_plan_json(
        '[{"title": "x", "tool": "web_search", "args": "not a dict"}]', TOOLS)
    assert steps[0].args == {}


# ── Building a plan ──────────────────────────────────────────────────────────

def test_build_plan_uses_the_planner_and_persists():
    plan = engine.build_plan(
        "compare three flights", TOOLS,
        planner=lambda prompt, system: '[{"title": "Search flights",'
                                       ' "tool": "web_search", "args": {"query": "flights"}}]')

    assert plan.id
    assert plan.status == engine.PLANNED
    assert len(plan.steps) == 1
    assert engine.load(plan.id).goal == "compare three flights"


def test_the_planner_is_told_which_tools_exist():
    seen = {}

    def planner(prompt, system):
        seen["prompt"] = prompt
        return "[]"

    engine.build_plan("do a thing", {"web_search", "file_processor"}, planner=planner)

    assert "- web_search" in seen["prompt"]
    assert "- file_processor" in seen["prompt"]
    assert "- think" in seen["prompt"]


def test_the_planner_is_told_not_to_plan_destructive_steps():
    seen = {}

    def planner(prompt, system):
        seen["prompt"] = prompt
        return "[]"

    engine.build_plan("clean up my disk", TOOLS, planner=planner)

    assert "Do not plan anything destructive" in seen["prompt"]


def test_a_failed_planner_still_produces_a_usable_plan():
    """No plan at all is a dead end for the user; one 'think' step means the
    assistant answers the goal directly instead of reporting a planning error."""
    def boom(prompt, system):
        raise RuntimeError("every model on the ladder failed")

    plan = engine.build_plan("explain heat pumps", TOOLS, planner=boom)

    assert len(plan.steps) == 1
    assert plan.steps[0].tool == "think"
    assert plan.steps[0].args["question"] == "explain heat pumps"


def test_an_empty_plan_falls_back_the_same_way():
    plan = engine.build_plan("hello", TOOLS, planner=lambda p, s: "[]")
    assert [s.tool for s in plan.steps] == ["think"]


def test_the_goal_is_length_capped():
    plan = engine.build_plan("x" * 5000, TOOLS, planner=lambda p, s: "[]")
    assert len(plan.goal) == engine.GOAL_MAX_CHARS


# ── Execution bookkeeping ────────────────────────────────────────────────────

def _plan(n=3):
    return engine.build_plan(
        "a goal", TOOLS,
        planner=lambda p, s: json.dumps(
            [{"title": f"step {i}", "tool": "web_search"} for i in range(n)]))


def test_start_step_marks_it_running_and_counts_the_attempt():
    plan = _plan()
    step = engine.start_step(plan.id, 0)

    assert step.status == engine.RUNNING
    assert step.attempts == 1
    assert engine.load(plan.id).status == engine.IN_PROGRESS


def test_a_successful_step_completes_the_plan_when_it_is_the_last():
    plan = _plan(1)
    engine.start_step(plan.id, 0)
    updated = engine.record_result(plan.id, 0, ok=True, result="found three")

    assert updated.status == engine.COMPLETED
    assert updated.steps[0].result == "found three"


def test_progress_is_counted():
    plan = _plan(3)
    for i in range(2):
        engine.start_step(plan.id, i)
        engine.record_result(plan.id, i, ok=True, result="ok")

    assert engine.load(plan.id).done_count() == 2
    assert engine.load(plan.id).next_pending().title == "step 2"


def test_a_failed_step_may_be_retried_up_to_the_bound():
    plan = _plan(2)

    engine.start_step(plan.id, 0)
    engine.record_result(plan.id, 0, ok=False, error="timeout")
    assert engine.can_retry(plan.id, 0) is True

    engine.start_step(plan.id, 0)
    engine.record_result(plan.id, 0, ok=False, error="timeout again")

    assert engine.can_retry(plan.id, 0) is False
    assert engine.load(plan.id).status == engine.ABANDONED


def test_a_plan_stops_when_a_step_runs_out_of_attempts():
    """Limping on past a failed dependency and reporting success is worse than
    stopping and saying so."""
    plan = _plan(3)
    for _ in range(engine.MAX_ATTEMPTS):
        engine.start_step(plan.id, 0)
        engine.record_result(plan.id, 0, ok=False, error="no")

    assert engine.load(plan.id).status == engine.ABANDONED
    assert engine.load(plan.id).next_pending().title == "step 1"


def test_skipping_a_step_lets_the_plan_finish():
    plan = _plan(2)
    engine.skip_step(plan.id, 0, "not needed")
    engine.start_step(plan.id, 1)
    updated = engine.record_result(plan.id, 1, ok=True, result="done")

    assert updated.status == engine.COMPLETED


def test_cancel():
    plan = _plan()
    assert engine.cancel(plan.id) is True
    assert engine.load(plan.id).status == engine.CANCELLED
    assert engine.load(plan.id).is_finished() is True


def test_long_results_are_truncated_on_the_way_in():
    plan = _plan(1)
    engine.start_step(plan.id, 0)
    engine.record_result(plan.id, 0, ok=True, result="x" * 5000)

    assert len(engine.load(plan.id).steps[0].result) == engine.STEP_RESULT_CHARS


def test_unknown_ids_are_handled_quietly():
    assert engine.start_step("nope", 0) is None
    assert engine.record_result("nope", 0, ok=True) is None
    assert engine.cancel("nope") is False
    assert engine.load("nope") is None


# ── Persistence ──────────────────────────────────────────────────────────────

def test_a_plan_survives_a_restart():
    plan = _plan(2)
    engine.start_step(plan.id, 0)
    engine.record_result(plan.id, 0, ok=True, result="first one done")

    reloaded = engine.load(plan.id)

    assert reloaded.steps[0].status == engine.DONE
    assert reloaded.steps[0].result == "first one done"
    assert reloaded.next_pending().title == "step 1"
    assert engine.active().id == plan.id


def test_active_returns_the_unfinished_plan():
    done = _plan(1)
    engine.start_step(done.id, 0)
    engine.record_result(done.id, 0, ok=True, result="ok")
    open_plan = _plan(2)

    assert engine.active().id == open_plan.id


def test_finished_plans_are_not_offered_as_the_active_one():
    plan = _plan(1)
    engine.start_step(plan.id, 0)
    engine.record_result(plan.id, 0, ok=True, result="ok")

    assert engine.active() is None


def test_the_store_does_not_grow_without_bound():
    for i in range(engine.MAX_PLANS_KEPT + 10):
        engine.build_plan(f"goal {i}", TOOLS, planner=lambda p, s: "[]")

    assert len(engine.load_all()) == engine.MAX_PLANS_KEPT


def test_a_corrupt_store_is_rebuilt_rather_than_fatal():
    plan = _plan(1)
    engine.STORE_PATH.write_text("{not json", encoding="utf-8")

    assert engine.load_all() == []
    assert engine.load(plan.id) is None
    engine.build_plan("new", TOOLS, planner=lambda p, s: "[]")
    assert len(engine.load_all()) == 1


# ── The safety line ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("tool", sorted(engine.DEFAULT_ALLOWED_TOOLS))
def test_allowed_tools_run_without_asking(tool):
    step = engine.TaskStep(title="x", tool=tool)
    assert step.needs_confirmation() is False


@pytest.mark.parametrize("tool", [
    "file_controller", "computer_settings", "computer_control", "desktop",
    "send_message", "dev_agent", "browser_control", "reminder",
])
def test_anything_else_needs_confirmation(tool):
    """The rule is an allow-list, not a blocklist: a new action that writes or
    sends is gated by default, and does not have to be remembered by anyone."""
    step = engine.TaskStep(title="x", tool=tool)
    assert step.needs_confirmation() is True


def test_render_flags_a_step_that_will_ask():
    plan = engine.build_plan(
        "tidy the desktop", TOOLS,
        planner=lambda p, s: '[{"title": "Move the files", "tool": "file_controller"}]')

    out = engine.render(plan)

    assert "needs your confirmation" in out


# ── Reporting ────────────────────────────────────────────────────────────────

def test_summary_is_one_line():
    plan = _plan(3)
    assert "\n" not in plan.summary()


def test_render_lists_every_step_with_its_mark():
    plan = _plan(2)
    engine.start_step(plan.id, 0)
    engine.record_result(plan.id, 0, ok=True, result="found it")

    out = engine.render(engine.load(plan.id))

    assert "Goal: a goal" in out
    assert "✓ step 0 — found it" in out
    assert "· step 1" in out


def test_list_plans_mentions_each_one():
    engine.build_plan("first goal", TOOLS, planner=lambda p, s: "[]")
    engine.build_plan("second goal", TOOLS, planner=lambda p, s: "[]")

    out = engine.list_plans()

    assert "first goal" in out and "second goal" in out


def test_list_plans_with_nothing_stored():
    assert engine.list_plans() == "No plans yet."
