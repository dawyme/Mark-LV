"""The task engine: planning, bounds, persistence, and the safety line.

A plan is the one place in this app where a model gets to decide what happens
next, more than once, without a human in the loop for each step. These tests
are mostly about the edges of that: what a malformed plan does, how many times
a failing step may be retried, and which steps are allowed to run without
somebody pressing a button.
"""
import json
import threading

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


def test_build_plan_can_resume_a_pre_reserved_persistent_id():
    plan = engine.build_plan("search", TOOLS,
                             planner=lambda *_: "[]", plan_id="reserved-job-plan")

    assert plan.id == "reserved-job-plan"
    assert engine.load("reserved-job-plan").goal == "search"


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

    assert "Never plan deletion, moving/renaming, overwriting" in seen["prompt"]


def test_failed_suffix_can_be_replanned_without_losing_attempt_history():
    plan = engine.build_plan(
        "find and summarize", TOOLS,
        planner=lambda _prompt, _system: json.dumps([
            {"title": "Completed lookup", "tool": "web_search"},
            {"title": "Failed source", "tool": "web_search"},
            {"title": "Old summary", "tool": "think"},
        ]),
    )
    engine.start_step(plan.id, 0)
    engine.record_result(plan.id, 0, True, result="PRIVATE local file contents")
    for _ in range(engine.MAX_ATTEMPTS):
        engine.start_step(plan.id, 1)
        engine.record_result(plan.id, 1, False, error="source timed out")

    captured = {}

    def replan(prompt, _system):
        captured["prompt"] = prompt
        return ('[{"title":"Use alternative source",'
                '"tool":"file_processor","args":{"file_path":"brief.txt"}}]')

    replacement = engine.replan_remaining(plan.id, 1, TOOLS, planner=replan)

    assert "PRIVATE local file contents" not in captured["prompt"]
    assert replacement is not None
    assert replacement.status == engine.IN_PROGRESS
    assert [(s.title, s.status) for s in replacement.steps] == [
        ("Completed lookup", engine.DONE), ("Use alternative source", engine.PENDING)
    ]
    history = replacement.history[-1]["steps"]
    assert history[0]["attempts"] == engine.MAX_ATTEMPTS
    assert history[0]["error"] == "source timed out"
    assert history[1]["title"] == "Old summary"


def test_replan_rejects_an_invalid_or_empty_alternative_without_changing_plan():
    plan = engine.build_plan("goal", TOOLS,
                             planner=lambda *_: '[{"title":"broken","tool":"web_search"}]')
    engine.start_step(plan.id, 0)
    engine.record_result(plan.id, 0, False, error="failed")
    engine.start_step(plan.id, 0)
    engine.record_result(plan.id, 0, False, error="still failed")
    before = engine.load(plan.id)

    replanned = engine.replan_remaining(plan.id, 0, TOOLS,
                                        planner=lambda *_: "[]")

    assert replanned is None
    assert engine.load(plan.id).steps[0].title == before.steps[0].title
    assert not engine.load(plan.id).history


def test_corrupt_task_store_is_preserved_before_new_writes():
    engine.STORE_PATH.write_text('{partial json', encoding="utf-8")

    assert engine.load_all() == []
    assert not engine.STORE_PATH.exists()
    assert engine.STORE_PATH.with_suffix(engine.STORE_PATH.suffix + ".corrupt").exists()


def test_valid_task_history_survives_an_invalid_partial_row():
    valid = {
        "id": "kept", "goal": "keep me", "created": "now", "status": engine.PLANNED,
        "steps": [{"title": "read", "tool": "web_search", "status": engine.PENDING}],
    }
    engine.STORE_PATH.write_text(json.dumps({"plans": [valid, None, {"id": "bad"}]}),
                                 encoding="utf-8")

    plans = engine.load_all()

    assert [plan.id for plan in plans] == ["kept"]
    assert engine.load("kept").goal == "keep me"


def test_planner_receives_argument_level_file_and_code_safety_rules():
    prompts = {}

    def planner(prompt, _system):
        prompts["prompt"] = prompt
        return "[]"

    engine.build_plan("summarize my report", TOOLS, planner=planner)
    prompt = prompts["prompt"]

    assert '"save": false' in prompt
    assert 'action: "explain"' in prompt
    assert "`weather_report` and" in prompt
    assert "`flight_finder` open a browser" in prompt


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
    assert engine.active_step_indices(plan.id) == {0}

    engine.record_result(plan.id, 0, ok=True, result="finished")
    assert engine.active_step_indices(plan.id) == set()


def test_parallel_callers_cannot_claim_the_same_step_twice():
    plan = _plan(1)
    barrier = threading.Barrier(3)
    claimed = []

    def claim():
        barrier.wait()
        claimed.append(engine.start_step(plan.id, 0))

    threads = [threading.Thread(target=claim) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=2)

    assert sum(step is not None for step in claimed) == 1
    assert engine.load(plan.id).steps[0].attempts == 1


def test_reviewed_step_can_be_claimed_after_the_automatic_retry_budget():
    plan = _plan(1)
    plan.max_attempts = 1
    plan.status = engine.IN_PROGRESS
    plan.steps[0].status = engine.FAILED
    plan.steps[0].attempts = 1
    plan.steps[0].review_required = True
    engine._save(plan)

    restored = engine.load(plan.id)

    assert restored.next_pending() is not None
    claimed = engine.start_step(plan.id, 0)
    assert claimed is not None
    assert claimed.attempts == 2


def test_exhausted_step_without_review_cannot_be_claimed_again():
    plan = _plan(1)
    plan.max_attempts = 1
    plan.steps[0].status = engine.FAILED
    plan.steps[0].attempts = 1
    plan.status = engine.IN_PROGRESS
    engine._save(plan)

    assert engine.load(plan.id).next_pending() is None
    assert engine.start_step(plan.id, 0) is None


def test_steps_cannot_be_claimed_out_of_order():
    plan = _plan(2)

    assert engine.start_step(plan.id, 1) is None
    assert engine.load(plan.id).steps[1].status == engine.PENDING


def test_repaired_arguments_are_atomically_saved_only_after_an_attempt_finishes():
    plan = _plan(1)
    engine.start_step(plan.id, 0)

    assert engine.update_step_args(plan.id, 0, {"query": "fixed"}) is None
    engine.record_result(plan.id, 0, ok=False, error="bad query")
    updated = engine.update_step_args(plan.id, 0, {"query": "fixed"},
                                      review_required=True,
                                      error="review the repaired arguments")

    assert updated.steps[0].args == {"query": "fixed"}
    assert updated.steps[0].review_required is True
    assert updated.steps[0].error == "review the repaired arguments"
    assert engine.load(plan.id).steps[0].args == {"query": "fixed"}


def test_interrupted_review_flag_is_persistent_and_requires_approval():
    plan = _plan(1)
    plan.steps[0].review_required = True
    engine._save(plan)

    restored = engine.load(plan.id)

    assert restored.steps[0].review_required
    assert restored.steps[0].needs_confirmation()


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


def test_action_result_does_not_revive_a_plan_cancelled_in_flight():
    plan = _plan(1)
    engine.start_step(plan.id, 0)
    assert engine.cancel(plan.id)

    updated = engine.record_result(plan.id, 0, ok=True, result="finished before stop")

    assert updated.status == engine.CANCELLED
    assert updated.steps[0].status == engine.DONE


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


def test_the_store_keeps_a_bounded_finished_plan_history():
    for i in range(engine.MAX_PLANS_KEPT + 10):
        plan = engine.build_plan(f"goal {i}", TOOLS, planner=lambda p, s: "[]")
        engine.start_step(plan.id, 0)
        engine.record_result(plan.id, 0, ok=True, result="complete")

    assert len(engine.load_all()) == engine.MAX_PLANS_KEPT


def test_finished_plan_retention_never_evicts_unfinished_work():
    active = _plan(1)
    for i in range(engine.MAX_PLANS_KEPT + 10):
        plan = engine.build_plan(f"finished goal {i}", TOOLS,
                                 planner=lambda p, s: "[]")
        engine.start_step(plan.id, 0)
        engine.record_result(plan.id, 0, ok=True, result="complete")

    assert engine.load(active.id) is not None
    plans = engine.load_all()
    assert sum(not plan.is_finished() for plan in plans) == 1
    assert sum(plan.is_finished() for plan in plans) == engine.MAX_PLANS_KEPT - 1


def test_a_corrupt_store_is_rebuilt_rather_than_fatal():
    plan = _plan(1)
    engine.STORE_PATH.write_text("{not json", encoding="utf-8")

    assert engine.load_all() == []
    assert engine.load(plan.id) is None
    engine.build_plan("new", TOOLS, planner=lambda p, s: "[]")
    assert len(engine.load_all()) == 1


# ── The safety line ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("tool", sorted(engine.DEFAULT_ALLOWED_TOOLS - {
    "file_controller", "file_processor", "computer_settings", "open_app",
    "code_helper", "flight_finder"
}))
def test_simple_read_only_allowed_tools_run_without_asking(tool):
    step = engine.TaskStep(title="x", tool=tool)
    assert step.needs_confirmation() is False


def test_browser_based_flight_search_requires_confirmation():
    step = engine.TaskStep(title="search flights", tool="flight_finder",
                           args={"origin": "POS", "destination": "LHR", "date": "2026-11-01"})
    assert step.needs_confirmation() is True


def test_flight_search_fails_closed_if_the_policy_module_cannot_load(monkeypatch):
    import builtins
    original_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "core.autonomy":
            raise ImportError("simulated policy load failure")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    step = engine.TaskStep(title="search flights", tool="flight_finder")

    assert step.needs_confirmation() is True


def test_broad_actions_require_safe_args_even_when_tool_is_known():
    assert engine.TaskStep(title="read", tool="file_processor",
                           args={"action": "info", "save": False}).needs_confirmation() is False
    assert engine.TaskStep(title="convert", tool="file_processor",
                           args={"action": "convert", "save": False}).needs_confirmation() is True
    assert engine.TaskStep(title="explain", tool="code_helper",
                           args={"action": "explain"}).needs_confirmation() is False
    assert engine.TaskStep(title="execute", tool="code_helper",
                           args={"action": "run"}).needs_confirmation() is True


@pytest.mark.parametrize("tool", [
    "file_controller", "computer_settings", "computer_control", "desktop",
    "send_message", "open_app", "dev_agent", "browser_control", "reminder",
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
