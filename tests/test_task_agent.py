"""The task_agent action: dispatch, retries, and the confirmation gate.

The property under test here is the one that is easy to get wrong: a plan may
not become a way around the safeguard that exists for a single action. A step
that deletes a file has to end up behind the same on-screen button as a
one-shot delete would, and the button has to be the thing that lets it run.
"""
import json

import pytest

from core import action_loader, confirm as confirm_gate, task_engine as engine
from actions import task_agent


class FakeRegistry:
    """Stands in for the live ActionRegistry.

    `outcomes` maps a tool name to a list of results, consumed in order, so a
    test can make the first attempt fail and the second succeed.
    """

    def __init__(self, names, outcomes=None):
        self._names    = set(names)
        self._outcomes = {k: list(v) for k, v in (outcomes or {}).items()}
        self.calls     = []

    def names(self):
        return set(self._names)

    def has(self, name):
        return name in self._names

    def run(self, name, parameters, ctx=None):
        self.calls.append({"name": name, "args": dict(parameters or {}),
                           "ctx": ctx})
        queue = self._outcomes.get(name)
        if queue:
            return queue.pop(0)
        return f"{name} did its job."


class FakePlayer:
    def __init__(self):
        self.logs     = []
        self.contents = []

    def write_log(self, text):
        self.logs.append(text)

    def show_content(self, label, text):
        self.contents.append((label, text))


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    engine.set_store_path(tmp_path / "tasks.json")
    from core import background_worker
    from core.autonomy import AutonomyPolicy

    class FakeWorker:
        def __init__(self):
            self.policy = AutonomyPolicy()
            self.started = False
            self.submissions = []

        def start(self):
            self.started = True
            return self

        def submit(self, goal, plan_id="", **kwargs):
            self.submissions.append((goal, plan_id))
            return background_worker.Job(id="jobtest", goal=goal, plan_id=plan_id,
                                         status=background_worker.QUEUED)

    fake = FakeWorker()
    monkeypatch.setattr(background_worker, "get_worker", lambda: fake)
    yield fake
    engine.set_store_path(engine._base_dir() / "memory" / "tasks.json")


@pytest.fixture
def registry(monkeypatch):
    def _install(names, outcomes=None):
        reg = FakeRegistry(names, outcomes)
        monkeypatch.setattr(action_loader, "get_registry", lambda: reg)
        return reg
    return _install


@pytest.fixture
def no_model(monkeypatch):
    """Keep the planner and the recovery step off the network."""
    from core import gemini
    monkeypatch.setattr(gemini, "text", lambda *a, **kw: "[]")
    monkeypatch.setattr(gemini, "as_json", lambda *a, **kw: None)


@pytest.fixture
def plan_of(monkeypatch, no_model):
    """Create a plan with these exact steps, and make `action=plan` remake it.

    Creating it immediately is the point: most of these tests are about
    running a step, not about planning one, and a planner that returns "[]"
    would otherwise leave them with nothing to run.
    """
    def _make(steps):
        raw = json.dumps(steps)

        def build(goal, tools, planner=None, max_steps=6):
            # Every tool the test named counts as available, or the engine
            # would drop the destructive steps these tests exist to check.
            known = {row.get("tool") for row in steps} | set(tools) | {"think"}
            plan = engine.TaskPlan(id="plan1", goal=goal or "a goal",
                                   created="now",
                                   steps=engine.parse_plan_json(raw, known))
            engine._save(plan)
            return plan

        monkeypatch.setattr(task_agent.engine, "build_plan", build)
        return build("a goal", {"web_search"})
    return _make


# ── Planning ─────────────────────────────────────────────────────────────────

def test_plan_queues_goal_without_planning_on_the_voice_path(registry, store, monkeypatch):
    registry({"web_search"})

    def planning_must_be_deferred(*_args, **_kwargs):
        raise AssertionError("the voice/UI path must not call the model planner")

    monkeypatch.setattr(engine, "build_plan", planning_must_be_deferred)

    out = task_agent.task_agent({"action": "plan", "goal": "compare flights"})

    assert "Started background job jobtest" in out
    assert "plan and execute this goal" in out
    assert store.submissions == [("compare flights", "")]
    assert store.started


def test_plan_without_a_goal_asks_for_one():
    out = task_agent.task_agent({"action": "plan"})
    assert "Tell me what you want done" in out


def test_disabled_autonomy_is_not_reported_as_automatic_execution(registry, store):
    registry({"web_search"})
    store.policy.profile.allow_autonomous = False

    out = task_agent.task_agent({"action": "plan", "goal": "search the web"})

    assert "Started background job jobtest" in out
    assert "Automatic execution is disabled" in out
    assert "Routine steps will run automatically" not in out


def test_disabled_background_profile_keeps_single_think_direct_answer_flow(
        plan_of, registry, store, monkeypatch):
    registry({"web_search"})
    store.policy.profile.enable_background_jobs = False
    plan_of([{"title": "Answer directly", "tool": "think",
              "args": {"question": "why is the sky blue"}}])
    monkeypatch.setattr(task_agent, "_think", lambda _question: "Direct answer")

    out = task_agent.task_agent({"action": "plan", "goal": "why is the sky blue"})

    assert out == "Direct answer"
    assert engine.load("plan1").status == engine.COMPLETED
    assert store.submissions == []


def test_single_think_step_is_queued_like_other_background_work(registry, store):
    """Even model-only work gets an auditable job and does not block voice/UI."""
    registry({"web_search"})

    out = task_agent.task_agent({"action": "plan", "goal": "why is the sky blue"})

    assert "Started background job" in out
    assert store.submissions == [("why is the sky blue", "")]


# ── Running a step ───────────────────────────────────────────────────────────

def test_run_executes_the_next_step_through_the_registry(plan_of, registry):
    reg = registry({"web_search"})
    plan_of([{"title": "Find flights", "tool": "web_search",
              "args": {"query": "LGW to POS"}}])

    out = task_agent.task_agent({"action": "run"})

    assert [c["name"] for c in reg.calls] == ["web_search"]
    assert reg.calls[0]["args"] == {"query": "LGW to POS"}
    # A one-step plan is reported as the answer, not as a progress line.
    assert "did its job" in out
    assert engine.load("plan1").status == engine.COMPLETED


def test_the_step_gets_the_player_so_it_can_log(plan_of, registry):
    reg = registry({"web_search"})
    plan_of([{"title": "Search", "tool": "web_search"}])
    player = FakePlayer()

    task_agent.task_agent({"action": "run"}, player=player)

    assert reg.calls[0]["ctx"]["player"] is player
    assert any("plan1" in line for line in player.logs)


def test_run_with_no_plan_says_so():
    assert "no plan" in task_agent.task_agent({"action": "run"}).lower()


def test_an_unknown_tool_in_a_step_is_reported_not_raised(plan_of, registry):
    plan_of([{"title": "Odd", "tool": "web_search"}])
    # The registry that is live has never heard of the tool the plan named.
    registry(set())

    out = task_agent.task_agent({"action": "run"})

    assert "did not work" in out or "no tool" in out.lower()


# ── Retry and recovery ───────────────────────────────────────────────────────

def test_a_failed_step_is_retried_and_can_succeed(plan_of, registry, monkeypatch):
    reg = registry({"web_search"},
                   outcomes={"web_search": ["Tool 'web_search' failed: timeout",
                                            "three flights found"]})
    plan_of([{"title": "Find flights", "tool": "web_search"}])
    monkeypatch.setattr(task_agent, "_recovered_args", lambda *a, **kw: {})

    out = task_agent.task_agent({"action": "run"})

    assert len(reg.calls) == 2
    assert "three flights found" in out
    assert engine.load("plan1").status == engine.COMPLETED


def test_recovery_asks_the_model_for_corrected_arguments(plan_of, registry,
                                                          monkeypatch):
    reg = registry({"web_search"},
                   outcomes={"web_search": ["Tool 'web_search' failed: bad query",
                                            "found it"]})
    plan_of([{"title": "Find flights", "tool": "web_search", "args": {"query": ""}}])
    monkeypatch.setattr(task_agent, "_recovered_args",
                        lambda tool, step, err: {"query": "LGW to POS"})

    task_agent.task_agent({"action": "run"})

    assert reg.calls[0]["args"] == {"query": ""}
    assert reg.calls[1]["args"] == {"query": "LGW to POS"}


def test_argument_repair_cannot_escalate_a_read_into_a_write(plan_of, registry, monkeypatch):
    reg = registry({"file_processor"}, outcomes={
        "file_processor": ["Could not read file: missing file"],
    })
    plan_of([{"title": "Inspect file", "tool": "file_processor", "args": {
        "file_path": "missing.txt", "action": "info", "save": False,
    }}])
    monkeypatch.setattr(task_agent, "_recovered_args", lambda *a, **kw: {
        "file_path": "missing.txt", "action": "convert", "save": False,
    })

    result = task_agent._execute_step("plan1", 0)
    step = engine.load("plan1").steps[0]

    assert len(reg.calls) == 1
    assert step.status == engine.FAILED
    assert step.args["action"] == "convert"
    assert step.review_required is True
    assert "confirmation" in result.lower()


def test_a_step_gives_up_after_the_bound(plan_of, registry, monkeypatch):
    reg = registry({"web_search"},
                   outcomes={"web_search": ["Tool 'web_search' failed: down"] * 5})
    plan_of([{"title": "Find flights", "tool": "web_search"}])
    monkeypatch.setattr(task_agent, "_recovered_args", lambda *a, **kw: {})

    out = task_agent.task_agent({"action": "run"})

    assert len(reg.calls) == engine.MAX_ATTEMPTS
    assert "did not work" in out
    assert engine.load("plan1").status == engine.ABANDONED


def test_an_empty_result_counts_as_a_failure(plan_of, registry, monkeypatch):
    reg = registry({"web_search"}, outcomes={"web_search": ["", ""]})
    plan_of([{"title": "Search", "tool": "web_search"}])
    monkeypatch.setattr(task_agent, "_recovered_args", lambda *a, **kw: {})

    task_agent.task_agent({"action": "run"})

    assert len(reg.calls) == engine.MAX_ATTEMPTS


# ── The confirmation gate ────────────────────────────────────────────────────

@pytest.fixture
def confirm_ui(monkeypatch):
    """Bind core.confirm to a fake HUD, and record what it would have shown."""
    shown = {}
    monkeypatch.setattr(confirm_gate, "_show_cb",
                        lambda title, detail: shown.update(title=title, detail=detail))
    monkeypatch.setattr(confirm_gate, "_hide_cb", lambda: None)
    monkeypatch.setattr(confirm_gate, "_log_cb", lambda msg: None)
    monkeypatch.setattr(confirm_gate, "_pending", None)
    return shown


def test_a_destructive_step_does_not_run(plan_of, registry, confirm_ui):
    """The whole point: a plan cannot reach a destructive tool by decomposition."""
    reg = registry({"file_controller"})
    plan_of([{"title": "Delete the old files", "tool": "file_controller",
              "args": {"action": "delete"}}])

    out = task_agent.task_agent({"action": "run"})

    assert reg.calls == []
    assert "[CONFIRMATION_PENDING]" in out
    assert "file_controller" in confirm_ui["title"]


def test_the_step_runs_only_when_the_user_confirms(plan_of, registry, confirm_ui):
    reg = registry({"file_controller"})
    plan_of([{"title": "Delete the old files", "tool": "file_controller",
              "args": {"action": "delete"}}])

    task_agent.task_agent({"action": "run"})
    assert reg.calls == []

    confirm_gate.resolve(accepted=True)
    import time
    time.sleep(0.2)                      # the gate runs it on a worker thread

    assert len(reg.calls) == 1
    assert engine.load("plan1").steps[0].status == engine.DONE


def test_cancelling_the_confirmation_does_nothing(plan_of, registry, confirm_ui):
    reg = registry({"file_controller"})
    plan_of([{"title": "Delete", "tool": "file_controller"}])

    task_agent.task_agent({"action": "run"})
    confirm_gate.resolve(accepted=False)
    import time
    time.sleep(0.2)

    assert reg.calls == []
    assert engine.load("plan1").steps[0].status != engine.DONE


def test_cancelled_plan_cannot_run_from_a_stale_confirmation(plan_of, registry, confirm_ui):
    reg = registry({"file_controller"})
    plan_of([{"title": "Delete", "tool": "file_controller",
              "args": {"action": "delete", "path": "old.txt"}}])

    task_agent.task_agent({"action": "run"})
    assert reg.calls == []
    engine.cancel("plan1")
    confirm_gate.resolve(accepted=True)
    import time
    time.sleep(0.2)

    assert reg.calls == []
    assert engine.load("plan1").status == engine.CANCELLED


def test_confirmation_does_not_apply_after_step_arguments_change(
        plan_of, registry, confirm_ui):
    reg = registry({"file_controller"})
    plan_of([{"title": "Delete", "tool": "file_controller",
              "args": {"action": "delete", "path": "old.txt"}}])

    task_agent.task_agent({"action": "run"})
    changed = engine.load("plan1")
    changed.steps[0].args["path"] = "new.txt"
    engine._save(changed)
    confirm_gate.resolve(accepted=True)
    import time
    time.sleep(0.2)

    assert reg.calls == []
    assert engine.load("plan1").steps[0].status == engine.PENDING


def test_uncertain_exhausted_step_waits_for_explicit_review(plan_of, registry, confirm_ui):
    reg = registry({"file_controller"})
    plan_of([{"title": "Delete a file", "tool": "file_controller",
              "args": {"action": "delete", "path": "old.txt"}}])
    plan = engine.load("plan1")
    plan.max_attempts = 1
    plan.steps[0].status = engine.FAILED
    plan.steps[0].attempts = 1
    plan.steps[0].review_required = True
    plan.status = engine.IN_PROGRESS
    engine._save(plan)

    out = task_agent.task_agent({"action": "run"})

    assert "[CONFIRMATION_PENDING]" in out
    assert "previous outcome is unknown" in confirm_ui["detail"]
    assert reg.calls == []
    confirm_gate.resolve(accepted=True)
    import time
    time.sleep(0.2)

    assert len(reg.calls) == 1
    assert engine.load("plan1").steps[0].status == engine.DONE
    assert engine.load("plan1").steps[0].review_required is False


def test_ui_approval_is_passed_to_the_computer_action_handler(
        plan_of, registry, confirm_ui):
    reg = registry({"computer_settings"})
    plan_of([{"title": "Shut down", "tool": "computer_settings",
              "args": {"action": "shutdown"}}])

    task_agent.task_agent({"action": "run"})
    assert reg.calls == []
    confirm_gate.resolve(accepted=True)
    import time
    time.sleep(0.2)

    assert len(reg.calls) == 1
    assert reg.calls[0]["ctx"]["_confirmation_approved"] is True
    assert engine.load("plan1").steps[0].status == engine.DONE


def test_confirmed_sensitive_step_is_not_retried_without_fresh_approval(
        plan_of, registry, confirm_ui, monkeypatch):
    reg = registry({"file_controller"}, outcomes={
        "file_controller": ["Tool 'file_controller' failed: uncertain outcome",
                            "unexpected second execution"]})
    plan_of([{"title": "Delete a file", "tool": "file_controller",
              "args": {"action": "delete", "path": "old.txt"}}])
    monkeypatch.setattr(task_agent, "_recovered_args",
                        lambda *a, **kw: {"action": "delete", "path": "other.txt"})

    task_agent.task_agent({"action": "run"})
    confirm_gate.resolve(accepted=True)
    import time
    time.sleep(0.2)

    assert len(reg.calls) == 1
    assert reg.calls[0]["args"] == {"action": "delete", "path": "old.txt"}
    assert engine.load("plan1").status == engine.IN_PROGRESS


def test_a_read_only_step_never_reaches_the_gate(plan_of, registry, confirm_ui):
    reg = registry({"web_search"})
    plan_of([{"title": "Search", "tool": "web_search"}])

    task_agent.task_agent({"action": "run"})

    assert len(reg.calls) == 1
    assert confirm_ui == {}


def test_file_controller_read_retains_the_existing_safe_default(plan_of, registry, confirm_ui):
    reg = registry({"file_controller"})
    plan_of([{"title": "List documents", "tool": "file_controller",
              "args": {"action": "list", "path": "documents"}}])

    task_agent.task_agent({"action": "run"})

    assert len(reg.calls) == 1
    assert reg.calls[0]["args"]["action"] == "list"
    assert confirm_ui == {}


def test_known_local_app_launch_retains_the_existing_safe_default(
        plan_of, registry, confirm_ui):
    reg = registry({"open_app"})
    plan_of([{"title": "Open calculator", "tool": "open_app",
              "args": {"app_name": "calculator"}}])

    task_agent.task_agent({"action": "run"})

    assert len(reg.calls) == 1
    assert confirm_ui == {}


def test_volume_adjustment_retains_the_existing_safe_default(plan_of, registry, confirm_ui):
    reg = registry({"computer_settings"})
    plan_of([{"title": "Lower the volume", "tool": "computer_settings",
              "args": {"action": "volume_down"}}])

    task_agent.task_agent({"action": "run"})

    assert len(reg.calls) == 1
    assert confirm_ui == {}


def test_file_creation_is_gated_by_the_central_write_permission(
        plan_of, registry, confirm_ui, store):
    reg = registry({"file_controller"})
    plan_of([{"title": "Create a report", "tool": "file_controller",
              "args": {"action": "create_file", "path": "documents",
                       "name": "report.txt", "content": "Summary"}}])

    out = task_agent.task_agent({"action": "run"})

    assert reg.calls == []
    assert confirm_ui
    assert "confirmation" in out.lower()


def test_file_creation_can_be_enabled_for_a_new_path(
        tmp_path, monkeypatch, plan_of, registry, confirm_ui, store):
    from actions import file_controller

    reg = registry({"file_controller"})
    monkeypatch.setattr(file_controller, "_resolve_path", lambda _raw: tmp_path)
    store.policy.profile.allow_file_writes = True
    plan_of([{"title": "Create a report", "tool": "file_controller",
              "args": {"action": "create_file", "path": str(tmp_path),
                       "name": "report.txt", "content": "Summary"}}])

    task_agent.task_agent({"action": "run"})

    assert len(reg.calls) == 1
    assert confirm_ui == {}


def test_two_banners_are_never_stacked(plan_of, registry, confirm_ui, monkeypatch):
    registry({"file_controller"})
    plan_of([{"title": "Delete", "tool": "file_controller"}])
    monkeypatch.setattr(confirm_gate, "pending_title", lambda: "Shutdown")

    out = task_agent.task_agent({"action": "run"})

    assert "waiting for your confirmation" in out


# ── finish / status / list / cancel ──────────────────────────────────────────

def test_finish_runs_every_remaining_step(plan_of, registry):
    reg = registry({"web_search"})
    plan_of([{"title": "one", "tool": "web_search"},
             {"title": "two", "tool": "web_search"}])

    out = task_agent.task_agent({"action": "finish"})

    assert len(reg.calls) == 2
    assert "All 2 steps" in out


def test_finish_stops_at_a_step_that_needs_confirmation(plan_of, registry,
                                                        confirm_ui):
    reg = registry({"web_search", "file_controller"})
    plan_of([{"title": "one", "tool": "web_search"},
             {"title": "two", "tool": "file_controller"}])

    out = task_agent.task_agent({"action": "finish"})

    assert len(reg.calls) == 1
    assert "confirmation" in out
    assert engine.load("plan1").status == engine.IN_PROGRESS


def test_finish_respects_the_wall_clock(plan_of, registry, monkeypatch):
    reg = registry({"web_search"})
    plan_of([{"title": str(i), "tool": "web_search"} for i in range(5)])
    clock = {"t": 0.0}
    monkeypatch.setattr(task_agent.time, "monotonic",
                        lambda: clock.__setitem__("t", clock["t"] + 100.0) or clock["t"])

    task_agent.task_agent({"action": "finish"})

    # Five steps planned; the budget cut it off long before the last one.
    assert 0 < len(reg.calls) < 5


def test_status_reports_progress(plan_of, registry):
    registry({"web_search"})
    plan_of([{"title": "one", "tool": "web_search"}])

    out = task_agent.task_agent({"action": "status"})

    assert "Goal:" in out and "Steps" in out


def test_status_with_no_plan():
    assert "No plan is running" in task_agent.task_agent({"action": "status"})


def test_list_and_cancel(plan_of, registry):
    registry({"web_search"})
    plan_of([{"title": "one", "tool": "web_search"}])
    task_agent.task_agent({"action": "plan", "goal": "a goal"})

    assert "a goal" in task_agent.task_agent({"action": "list"})

    out = task_agent.task_agent({"action": "cancel"})
    assert "Cancelled" in out
    assert engine.load("plan1").status == engine.CANCELLED


def test_an_unknown_action_is_answered_not_raised():
    assert "plan, run, finish, status, list" in task_agent.task_agent(
        {"action": "sideways"})


def test_action_result_failure_text_is_not_reported_as_success():
    assert task_agent._is_failed_result("Search failed: no network")
    assert task_agent._is_failed_result("Could not read file: permission denied")
    assert task_agent._is_failed_result("Please provide a search query.")
    assert task_agent._is_failed_result("Please specify width, height, or scale.")
    assert task_agent._is_failed_result("Pillow is not installed. Run: pip install Pillow")
    assert task_agent._is_failed_result("AI image analysis failed: provider unavailable")
    assert task_agent._is_failed_result("Video transcription failed: ffmpeg error")
    assert task_agent._is_failed_result("Unknown video action: 'summarize'.")
    assert task_agent._is_failed_result("I couldn't find any flights; the page may not have loaded")
    assert task_agent._is_failed_result("I could not match 'invented' to a computer action")
    assert task_agent._is_failed_result("Access denied: outside the permitted directory")
    assert task_agent._is_failed_result("Path not found: missing.txt")
    assert task_agent._is_failed_result("No application name provided.")
    assert task_agent._is_failed_result("Unsupported operating system: Plan 9")
    assert task_agent._is_failed_result("Done.")
    assert not task_agent._is_failed_result("No results found for: a rare query")
    assert not task_agent._is_failed_result("Found three useful results")


def test_file_processor_preserves_empty_results_for_task_verification(tmp_path, monkeypatch):
    from actions import file_processor
    path = tmp_path / "report.txt"
    path.write_text("contents", encoding="utf-8")
    monkeypatch.setattr(file_processor, "_process_text_doc", lambda *a, **kw: "")

    result = file_processor.file_processor({
        "file_path": str(path), "action": "info", "save": False,
    })

    assert result == ""


def test_a_crash_inside_the_action_is_spoken_not_raised(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("disk on fire")
    monkeypatch.setattr(task_agent.engine, "list_plans", boom)

    out = task_agent.task_agent({"action": "list"})

    assert "did not run" in out and "disk on fire" in out


# ── The declaration itself ───────────────────────────────────────────────────

def test_the_tool_declaration_is_valid_for_the_loader():
    """core.action_loader validates TOOL at startup; a malformed one is
    rejected there, silently, and the skill never appears."""
    from core.action_loader import _validate
    import types

    module = types.SimpleNamespace(TOOL=task_agent.TOOL)
    record = _validate(module, "task_agent.py")

    assert record.valid, record.error
    assert record.name == "task_agent"
    assert record.parameters["type"] == "OBJECT"


def test_the_declaration_tells_the_model_about_confirmation():
    """The model has to know a step may stop for a button, or it will report a
    destructive step as finished."""
    assert "confirm" in task_agent.TOOL["description"].lower()
