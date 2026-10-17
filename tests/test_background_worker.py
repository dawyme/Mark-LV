"""Offline tests for durable jobs, execution, recovery, policy, and shutdown."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict

import pytest

from actions import task_agent
from core import action_loader, autonomy, background_worker as bw, task_engine as engine


class FakeRegistry:
    def __init__(self, names, outcomes=None):
        self._names = set(names)
        self._outcomes = {key: list(values) for key, values in (outcomes or {}).items()}
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def names(self):
        return set(self._names)

    def has(self, name):
        return name in self._names

    def run(self, name, parameters, ctx=None):
        with self.lock:
            self.calls.append((name, dict(parameters or {})))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            queue = self._outcomes.get(name, [])
            outcome = queue.pop(0) if queue else f"verified {name} result"
        try:
            if callable(outcome):
                return outcome(name, parameters)
            return outcome
        finally:
            with self.lock:
                self.active -= 1


@pytest.fixture
def harness(tmp_path, monkeypatch):
    tasks_path = tmp_path / "tasks.json"
    jobs_path = tmp_path / "jobs.json"
    old_path = engine.STORE_PATH
    engine.set_store_path(tasks_path)
    monkeypatch.setattr(task_agent, "_recovered_args", lambda *a, **kw: {})

    state = {"rows": [{"title": "Search the web", "tool": "web_search",
                       "args": {"query": "test"}}]}
    original_build = engine.build_plan

    def deterministic_build(goal, tools, planner=None, max_steps=engine.MAX_STEPS, **kwargs):
        return original_build(goal, tools,
                              planner=lambda _prompt, _system: json.dumps(state["rows"]),
                              max_steps=max_steps, **kwargs)

    monkeypatch.setattr(engine, "build_plan", deterministic_build)
    registry = FakeRegistry({"web_search", "file_processor", "file_controller"})
    monkeypatch.setattr(action_loader, "get_registry", lambda: registry)
    policy = autonomy.AutonomyPolicy(autonomy.AutonomyProfile(retry_backoff_seconds=0))

    def worker(*, synchronous=True, max_workers=2, store=jobs_path, chosen_policy=policy):
        return bw.BackgroundWorker(store_path=store, synchronous=synchronous,
                                   max_workers=max_workers, poll_interval=0.05,
                                   autonomy_policy=chosen_policy)

    yield {"tasks": tasks_path, "jobs": jobs_path, "rows": state,
           "registry": registry, "policy": policy, "worker": worker,
           "original_build": original_build}
    engine.set_store_path(old_path)


def _make_plan(harness, goal="a goal", rows=None, tools=None):
    tools = tools or {"web_search", "file_processor", "file_controller"}
    rows = rows or [{"title": "Search", "tool": "web_search", "args": {"query": "x"}}]
    return harness["original_build"](goal, tools,
                                     planner=lambda _prompt, _system: json.dumps(rows))


def test_natural_language_plan_is_automatically_executed_without_run_command(harness, monkeypatch):
    rows = [
        {"title": "Search", "tool": "web_search", "args": {"query": "flights"}},
        {"title": "Read result", "tool": "file_processor", "args": {"file_path": "notes.txt", "action": "info", "save": False}},
    ]
    harness["rows"]["rows"] = rows
    worker = harness["worker"](synchronous=True)
    monkeypatch.setattr(bw, "get_worker", lambda: worker)

    response = task_agent.task_agent({"action": "plan", "goal": "compare flights"})

    jobs = worker.list_jobs()
    assert "Started background job" in response
    assert len(jobs) == 1
    assert jobs[0].status == bw.COMPLETED
    assert [name for name, _args in harness["registry"].calls] == ["web_search", "file_processor"]
    plan = engine.load(jobs[0].plan_id)
    assert plan.status == engine.COMPLETED
    assert all(step.status == engine.DONE for step in plan.steps)
    assert jobs[0].plan_snapshot["steps"][0]["status"] == engine.DONE


def test_worker_goal_submission_creates_a_plan_and_runs_it(harness):
    worker = harness["worker"](synchronous=True)

    job = worker.submit("search for a fact")

    assert job.status == bw.COMPLETED
    assert job.plan_id
    assert engine.load(job.plan_id).steps[0].status == engine.DONE


def test_submitting_the_same_plan_is_idempotent(harness):
    worker = harness["worker"](synchronous=True)
    plan = _make_plan(harness)

    first = worker.submit(plan.goal, plan_id=plan.id)
    calls_after_first = len(harness["registry"].calls)
    duplicate = worker.submit(plan.goal, plan_id=plan.id)

    assert duplicate.id == first.id
    assert len(harness["registry"].calls) == calls_after_first == 1


def test_interrupted_plan_creation_recovers_into_the_reserved_id(harness):
    store = bw.JobStore(harness["jobs"])
    store.save(bw.Job(
        id="planning1", goal="resume planning", plan_id="stable-plan-id",
        status=bw.RUNNING, metadata={"_plan_pending": True},
        created="2026-01-01 00:00:00"))
    worker = harness["worker"](synchronous=True)

    assert "planning1" in worker.recover()
    assert worker.resume("planning1")
    job = worker.get_job("planning1")

    assert job.status == bw.COMPLETED
    assert job.plan_id == "stable-plan-id"
    assert engine.load("stable-plan-id") is not None
    assert [plan.id for plan in engine.load_all()].count("stable-plan-id") == 1


def test_failed_step_retries_with_repaired_arguments(harness, monkeypatch):
    harness["rows"]["rows"] = [{"title": "Search", "tool": "web_search",
                                  "args": {"query": ""}}]
    harness["registry"]._outcomes["web_search"] = [
        "Tool 'web_search' failed: query is empty", "search results verified"
    ]
    monkeypatch.setattr(task_agent, "_recovered_args",
                        lambda _tool, _step, _error: {"query": "corrected query"})
    worker = harness["worker"](synchronous=True)

    job = worker.submit("search")

    assert job.status == bw.COMPLETED
    assert harness["registry"].calls == [
        ("web_search", {"query": ""}),
        ("web_search", {"query": "corrected query"}),
    ]
    step = engine.load(job.plan_id).steps[0]
    assert step.status == engine.DONE and step.attempts == 2


def test_worker_replans_once_with_a_registered_alternative_after_retries(harness, monkeypatch):
    plan = _make_plan(harness, goal="find and summarize", rows=[
        {"title": "Search primary source", "tool": "web_search", "args": {"query": "q"}},
    ])
    harness["registry"]._outcomes["web_search"] = [
        "Search failed: primary source offline", "Search failed: primary source offline"
    ]
    from core import gemini
    monkeypatch.setattr(gemini, "text", lambda *a, **kw: json.dumps([
        {"title": "Use a local source", "tool": "file_processor",
         "args": {"file_path": "brief.txt", "action": "info", "save": False}},
    ]))
    worker = harness["worker"](synchronous=True)

    job = worker.submit("find and summarize", plan_id=plan.id)

    assert job.status == bw.COMPLETED
    assert [name for name, _ in harness["registry"].calls] == [
        "web_search", "web_search", "file_processor"
    ]
    final_plan = engine.load(plan.id)
    assert final_plan.steps[-1].title == "Use a local source"
    assert final_plan.history[-1]["steps"][0]["attempts"] == 2
    assert job.metadata["replan_attempts"] == 1
    assert any("Alternative plan saved" in entry for entry in job.log)


def test_configured_retry_limit_is_applied_to_new_plans(harness):
    harness["registry"]._outcomes["web_search"] = ["Search failed: offline"]
    policy = autonomy.AutonomyPolicy(autonomy.AutonomyProfile(
        max_retries=0, retry_backoff_seconds=0))
    worker = harness["worker"](synchronous=True, chosen_policy=policy)

    job = worker.submit("search with no retries")

    assert job.status == bw.FAILED
    assert len(harness["registry"].calls) == 1
    assert engine.load(job.plan_id).max_attempts == 1


def test_job_and_plan_persist_across_a_simulated_restart(harness):
    plan = _make_plan(harness, rows=[
        {"title": "Already done", "tool": "web_search", "args": {"query": "done"}},
        {"title": "Next work", "tool": "file_processor", "args": {"file_path": "x.txt", "action": "info", "save": False}},
    ])
    engine.start_step(plan.id, 0)
    engine.record_result(plan.id, 0, True, result="cached, complete")
    job = bw.Job(id="restart1", goal=plan.goal, plan_id=plan.id,
                 plan_snapshot=asdict(engine.load(plan.id)), status=bw.RUNNING,
                 created="2026-01-01 00:00:00")
    first_store = bw.JobStore(harness["jobs"])
    first_store.save(job)

    restarted = harness["worker"](synchronous=True)
    assert "restart1" in restarted.recover()
    recovered = restarted.get_job("restart1")
    assert recovered.status == bw.QUEUED
    assert recovered.plan_snapshot["steps"][0]["status"] == engine.DONE
    assert restarted.resume("restart1")
    final = restarted.get_job("restart1")

    assert final.status == bw.COMPLETED
    assert [name for name, _ in harness["registry"].calls] == ["file_processor"]
    assert "cached, complete" in engine.load(plan.id).steps[0].result


def test_recovery_finalizes_a_plan_completed_before_its_job_row(harness):
    plan = _make_plan(harness)
    engine.start_step(plan.id, 0)
    engine.record_result(plan.id, 0, True, result="verified before crash")
    bw.JobStore(harness["jobs"]).save(bw.Job(
        id="finalize1", goal=plan.goal, plan_id=plan.id,
        plan_snapshot=asdict(engine.load(plan.id)), status=bw.RUNNING,
        created="2026-01-01 00:00:00"))
    worker = harness["worker"](synchronous=True)

    recovered = worker.recover()
    job = worker.get_job("finalize1")

    assert "finalize1" not in recovered
    assert job.status == bw.COMPLETED
    assert job.result == engine.load(plan.id).summary()
    assert engine.load(plan.id).status == engine.COMPLETED
    assert harness["registry"].calls == []


def test_recovery_honors_a_persisted_cancellation_request(harness):
    plan = _make_plan(harness)
    bw.JobStore(harness["jobs"]).save(bw.Job(
        id="cancelrecover1", goal=plan.goal, plan_id=plan.id,
        plan_snapshot=asdict(plan), status=bw.RUNNING, cancel_requested=True,
        created="2026-01-01 00:00:00"))
    worker = harness["worker"](synchronous=True)

    recovered = worker.recover()
    job = worker.get_job("cancelrecover1")

    assert "cancelrecover1" not in recovered
    assert job.status == bw.CANCELLED
    assert engine.load(plan.id).status == engine.CANCELLED
    assert harness["registry"].calls == []


def test_running_safe_step_is_recovered_and_successful_steps_are_not_repeated(harness):
    plan = _make_plan(harness, rows=[
        {"title": "Already done", "tool": "file_processor", "args": {"file_path": "done", "action": "info", "save": False}},
        {"title": "Interrupted", "tool": "web_search", "args": {"query": "retry"}},
    ])
    engine.start_step(plan.id, 0)
    engine.record_result(plan.id, 0, True, result="already complete")
    engine.start_step(plan.id, 1)  # simulate a crash inside the second tool
    engine._ACTIVE_STEP_CLAIMS.discard((plan.id, 1))  # simulate a fresh process
    snapshot = asdict(engine.load(plan.id))
    bw.JobStore(harness["jobs"]).save(bw.Job(
        id="crashjob", goal=plan.goal, plan_id=plan.id, plan_snapshot=snapshot,
        status=bw.RUNNING, created="2026-01-01 00:00:00"))

    worker = harness["worker"](synchronous=True)
    worker.recover()
    recovered_plan = engine.load(plan.id)
    assert recovered_plan.steps[0].status == engine.DONE
    assert recovered_plan.steps[1].status == engine.FAILED
    assert worker.resume("crashjob")

    assert worker.status("crashjob") == bw.COMPLETED
    assert [name for name, _ in harness["registry"].calls] == ["web_search"]
    completed_plan = engine.load(plan.id)
    assert completed_plan.steps[0].attempts == 1
    assert completed_plan.steps[1].attempts == 2


def test_recovery_ignores_a_live_manually_confirmed_step(harness):
    plan = _make_plan(harness, rows=[{
        "title": "Delete a file", "tool": "file_controller",
        "args": {"action": "delete", "path": "old.txt"},
    }])
    engine.start_step(plan.id, 0)
    job = bw.Job(id="manual-live1", goal=plan.goal, plan_id=plan.id,
                 plan_snapshot=asdict(engine.load(plan.id)), status=bw.WAITING,
                 created="2026-01-01 00:00:00")
    bw.JobStore(harness["jobs"]).save(job)
    worker = harness["worker"](synchronous=True)

    try:
        assert worker.recover() == []
        assert worker.get_job(job.id).status == bw.WAITING
        assert engine.load(plan.id).steps[0].status == engine.RUNNING
        assert harness["registry"].calls == []
    finally:
        engine._ACTIVE_STEP_CLAIMS.discard((plan.id, 0))


def test_waiting_job_recovers_a_crashed_confirmed_step_for_review(harness):
    plan = _make_plan(harness, rows=[{
        "title": "Delete a file", "tool": "file_controller",
        "args": {"action": "delete", "path": "old.txt"},
    }])
    engine.start_step(plan.id, 0)
    engine._ACTIVE_STEP_CLAIMS.discard((plan.id, 0))  # confirmed action died with app
    job = bw.Job(id="manual-crash1", goal=plan.goal, plan_id=plan.id,
                 plan_snapshot=asdict(engine.load(plan.id)), status=bw.WAITING,
                 created="2026-01-01 00:00:00")
    bw.JobStore(harness["jobs"]).save(job)
    worker = harness["worker"](synchronous=True)

    worker.recover()
    recovered_plan = engine.load(plan.id)

    assert worker.get_job(job.id).status == bw.WAITING
    assert recovered_plan.steps[0].status == engine.FAILED
    assert recovered_plan.steps[0].review_required is True
    assert harness["registry"].calls == []


def test_legacy_plan_with_an_interrupted_read_is_recovered(harness):
    plan = _make_plan(harness)
    engine.start_step(plan.id, 0)
    engine._ACTIVE_STEP_CLAIMS.discard((plan.id, 0))  # simulate a fresh process
    worker = harness["worker"](synchronous=True)

    recovered = worker.recover()
    job = next(job for job in worker.list_jobs() if job.plan_id == plan.id)

    assert job.id in recovered
    assert job.status == bw.QUEUED
    assert engine.load(plan.id).steps[0].status == engine.FAILED
    assert engine.load(plan.id).steps[0].review_required is False
    assert job.plan_snapshot["steps"][0]["status"] == engine.FAILED


def test_interrupted_non_replayable_step_waits_for_explicit_review(harness):
    plan = _make_plan(harness, rows=[{
        "title": "Change an external setting", "tool": "file_controller",
        "args": {"action": "delete", "path": "old.txt"},
    }])
    engine.start_step(plan.id, 0)
    engine._ACTIVE_STEP_CLAIMS.discard((plan.id, 0))  # simulate a fresh process
    job = bw.Job(id="uncertain1", goal=plan.goal, plan_id=plan.id,
                 plan_snapshot=asdict(engine.load(plan.id)), status=bw.RUNNING,
                 created="2026-01-01 00:00:00")
    bw.JobStore(harness["jobs"]).save(job)
    worker = harness["worker"](synchronous=True)

    worker.recover()
    recovered_plan = engine.load(plan.id)
    recovered_job = worker.get_job(job.id)

    assert recovered_job.status == bw.WAITING
    assert recovered_plan.steps[0].status == engine.FAILED
    assert recovered_plan.steps[0].review_required is True
    assert task_agent._needs_confirmation(recovered_plan.steps[0]) is True
    assert harness["registry"].calls == []
    assert worker.resume(job.id) is True
    assert worker.status(job.id) == bw.WAITING
    assert harness["registry"].calls == []


def test_interrupted_safe_step_at_retry_limit_fails_truthfully(harness):
    plan = _make_plan(harness)
    engine.start_step(plan.id, 0)
    engine._ACTIVE_STEP_CLAIMS.discard((plan.id, 0))  # simulate a fresh process
    plan = engine.load(plan.id)
    plan.max_attempts = 1
    engine._save(plan)
    bw.JobStore(harness["jobs"]).save(bw.Job(
        id="exhausted1", goal=plan.goal, plan_id=plan.id,
        plan_snapshot=asdict(engine.load(plan.id)), status=bw.RUNNING,
        created="2026-01-01 00:00:00"))
    worker = harness["worker"](synchronous=True)

    worker.recover()

    assert worker.status("exhausted1") == bw.FAILED
    assert "retry budget" in worker.get_job("exhausted1").error
    assert engine.load(plan.id).status == engine.ABANDONED
    assert harness["registry"].calls == []


def test_no_completed_step_can_be_started_twice(harness):
    plan = _make_plan(harness)
    engine.start_step(plan.id, 0)
    engine.record_result(plan.id, 0, True, result="done")

    assert engine.start_step(plan.id, 0) is None
    assert harness["registry"].calls == []


def test_policy_holds_unapproved_tools_in_waiting_state(harness):
    plan = _make_plan(harness, rows=[{"title": "Delete a file", "tool": "file_controller",
                             "args": {"action": "delete", "path": "x"}}])
    worker = harness["worker"](synchronous=True)

    job = worker.submit("delete a file", plan_id=plan.id)

    assert job.status == bw.WAITING
    assert "confirmation" in job.error
    assert harness["registry"].calls == []
    assert engine.load(plan.id).steps[0].status == engine.PENDING


def test_browser_based_flight_lookup_waits_for_confirmation(harness):
    plan = _make_plan(harness, rows=[{
        "title": "Search flights", "tool": "flight_finder",
        "args": {"origin": "POS", "destination": "LHR", "date": "2026-11-01"},
    }], tools={"web_search", "file_processor", "file_controller", "flight_finder"})
    worker = harness["worker"](synchronous=True)

    job = worker.submit("search flights", plan_id=plan.id)

    assert job.status == bw.WAITING
    assert "confirmation" in job.error
    assert harness["registry"].calls == []


def test_argument_repair_that_would_write_waits_for_confirmation(harness, monkeypatch):
    plan = _make_plan(harness, rows=[{
        "title": "Inspect a file", "tool": "file_processor",
        "args": {"file_path": "missing.txt", "action": "info", "save": False},
    }])
    harness["registry"]._outcomes["file_processor"] = ["File not found: missing.txt"]
    monkeypatch.setattr(task_agent, "_recovered_args", lambda *a, **kw: {
        "file_path": "missing.txt", "action": "convert", "save": False,
    })
    worker = harness["worker"](synchronous=True)

    job = worker.submit("inspect the file", plan_id=plan.id)
    step = engine.load(plan.id).steps[0]

    assert job.status == bw.WAITING
    assert step.status == engine.FAILED
    assert step.args["action"] == "convert"
    assert step.review_required is True
    assert len(harness["registry"].calls) == 1


def test_missing_model_provider_is_reported_as_failure_not_success(harness, monkeypatch):
    harness["rows"]["rows"] = [{"title": "Reason about the goal", "tool": "think",
                                  "args": {"question": "answer"}}]
    from core import gemini
    monkeypatch.setattr(gemini, "text", lambda *a, **kw: "")
    worker = harness["worker"](synchronous=True)

    job = worker.submit("answer without credentials")

    assert job.status == bw.FAILED
    assert "credentials" in job.error.lower() or "provider" in job.error.lower()
    assert engine.load(job.plan_id).status == engine.ABANDONED


def test_unavailable_tool_fails_truthfully(harness):
    plan = _make_plan(harness)
    harness["registry"]._names.remove("web_search")
    worker = harness["worker"](synchronous=True)

    job = worker.submit("search", plan_id=plan.id)

    assert job.status == bw.FAILED
    assert "no tool" in job.error.lower() or "not available" in job.error.lower()
    assert not harness["registry"].calls


def test_disabled_background_jobs_cannot_be_resumed(harness):
    plan = _make_plan(harness)
    job = bw.Job(id="disabled-resume1", goal=plan.goal, plan_id=plan.id,
                 plan_snapshot=asdict(plan), status=bw.WAITING,
                 created="2026-01-01 00:00:00")
    bw.JobStore(harness["jobs"]).save(job)
    disabled = autonomy.AutonomyPolicy(autonomy.AutonomyProfile(
        enable_background_jobs=False))
    worker = harness["worker"](synchronous=True, chosen_policy=disabled)

    assert worker.resume(job.id) is False
    assert worker.status(job.id) == bw.WAITING
    assert harness["registry"].calls == []


def test_cancel_stops_after_the_inflight_action_returns(harness):
    entered, release = threading.Event(), threading.Event()

    def blocked(_name, _args):
        entered.set()
        assert release.wait(3)
        return "read completed"

    harness["registry"]._outcomes["web_search"] = [blocked]
    harness["rows"]["rows"] = [
        {"title": "First", "tool": "web_search"},
        {"title": "Second", "tool": "file_processor", "args": {"action": "info", "file_path": "second.txt", "save": False}},
    ]
    worker = harness["worker"](synchronous=False, max_workers=1)
    try:
        job = worker.submit("two steps")
        assert entered.wait(2)
        assert worker.status(job.id) == bw.RUNNING
        assert worker.resume(job.id) is False
        assert worker.status(job.id) == bw.RUNNING
        assert worker.cancel(job.id)
        release.set()
        final = worker.wait_for(job.id, timeout=3)
        assert final is not None and final.status == bw.CANCELLED
        assert [name for name, _ in harness["registry"].calls] == ["web_search"]
    finally:
        release.set()
        worker.stop(timeout=3)


def test_cancel_after_the_final_action_preserves_completed_outcome(harness):
    entered, release = threading.Event(), threading.Event()

    def blocked_success(_name, _args):
        entered.set()
        assert release.wait(3)
        return "verified final result"

    harness["registry"]._outcomes["web_search"] = [blocked_success]
    harness["rows"]["rows"] = [
        {"title": "Search", "tool": "web_search", "args": {"query": "test"}},
    ]
    worker = harness["worker"](synchronous=False, max_workers=1)
    try:
        job = worker.submit("one final step")
        assert entered.wait(2)
        assert worker.cancel(job.id)
        release.set()
        final = worker.wait_for(job.id, timeout=3)

        assert final is not None and final.status == bw.COMPLETED
        assert engine.load(job.plan_id).status == engine.COMPLETED
        assert "after the final step completed" in final.log[-1]
    finally:
        release.set()
        worker.stop(timeout=3)


def test_cancel_during_a_failed_attempt_prevents_automatic_retry(harness):
    entered, release = threading.Event(), threading.Event()

    def blocked_failure(_name, _args):
        entered.set()
        assert release.wait(3)
        return "Search failed: temporary outage"

    harness["registry"]._outcomes["web_search"] = [blocked_failure, "unexpected retry"]
    harness["rows"]["rows"] = [
        {"title": "Search", "tool": "web_search", "args": {"query": "test"}},
    ]
    worker = harness["worker"](synchronous=False, max_workers=1)
    try:
        job = worker.submit("search once")
        assert entered.wait(2)
        assert worker.cancel(job.id)
        release.set()
        final = worker.wait_for(job.id, timeout=3)

        assert final is not None and final.status == bw.CANCELLED
        assert [name for name, _ in harness["registry"].calls] == ["web_search"]
        plan = engine.load(job.plan_id)
        assert plan.status == engine.CANCELLED
        assert plan.steps[0].attempts == 1
    finally:
        release.set()
        worker.stop(timeout=3)


def test_graceful_shutdown_preserves_progress_for_resume(harness):
    entered, release = threading.Event(), threading.Event()

    def blocked(_name, _args):
        entered.set()
        assert release.wait(3)
        return "step one done"

    harness["registry"]._outcomes["web_search"] = [blocked]
    harness["rows"]["rows"] = [
        {"title": "First", "tool": "web_search"},
        {"title": "Second", "tool": "file_processor", "args": {"action": "info", "file_path": "second.txt", "save": False}},
    ]
    worker = harness["worker"](synchronous=False, max_workers=1)
    try:
        job = worker.submit("continue after shutdown")
        assert entered.wait(2)
        job = worker.get_job(job.id)
        assert job.plan_id
        releaser = threading.Timer(0.1, release.set)
        releaser.start()
        worker.stop(timeout=2)
        releaser.join()
        parked = worker.wait_for(job.id, timeout=2)
        assert parked is not None and parked.status == bw.QUEUED
        assert engine.load(job.plan_id).steps[0].status == engine.DONE

        restarted = harness["worker"](synchronous=True)
        restarted.recover()
        restarted.resume(job.id)
        assert restarted.status(job.id) == bw.COMPLETED
        assert [name for name, _ in harness["registry"].calls] == ["web_search", "file_processor"]
    finally:
        release.set()
        worker.stop(timeout=3)


def test_multiple_jobs_run_within_configured_concurrency(harness):
    both_entered, release = threading.Event(), threading.Event()
    entry_lock = threading.Lock()
    entered_count = {"n": 0}

    def blocked(_name, _args):
        with entry_lock:
            entered_count["n"] += 1
            if entered_count["n"] >= 2:
                both_entered.set()
        assert release.wait(3)
        return "done"

    harness["registry"]._outcomes["web_search"] = [blocked, blocked]
    worker = harness["worker"](synchronous=False, max_workers=2)
    try:
        first = worker.submit("first goal")
        second = worker.submit("second goal")
        assert both_entered.wait(3)
        assert harness["registry"].max_active == 2
        release.set()
        assert worker.wait_for(first.id, timeout=3).status == bw.COMPLETED
        assert worker.wait_for(second.id, timeout=3).status == bw.COMPLETED
    finally:
        release.set()
        worker.stop(timeout=3)


def test_corrupt_job_store_is_preserved_before_recovery(harness):
    path = harness["jobs"]
    path.write_text('{ this is not valid json', encoding="utf-8")
    worker = harness["worker"](synchronous=True)

    assert worker.store.load_all() == []
    assert not path.exists()
    assert path.with_suffix(path.suffix + ".corrupt").exists()


def test_job_report_includes_status_steps_logs_and_blockers(harness):
    plan = _make_plan(harness)
    worker = harness["worker"](synchronous=True)
    job = worker.submit("report this", plan_id=plan.id)
    report = worker.report(job.id)

    assert f"Job {job.id}: completed" in report
    assert "Steps (1/1 done)" in report
    assert "All planned steps completed" in report


def test_concurrent_start_calls_create_only_one_scheduler(harness):
    worker = harness["worker"](synchronous=False)
    recoveries = []
    worker.recover = lambda: (time.sleep(0.03), recoveries.append(True), [])[2]
    start_gate = threading.Barrier(3)

    def start_together():
        start_gate.wait()
        worker.start()

    starters = [threading.Thread(target=start_together) for _ in range(2)]
    for thread in starters:
        thread.start()
    start_gate.wait()
    for thread in starters:
        thread.join(timeout=2)

    try:
        assert all(not thread.is_alive() for thread in starters)
        assert len(recoveries) == 1
        assert worker._thread is not None and worker._thread.is_alive()
    finally:
        worker.stop(timeout=1)


def test_worker_start_is_idempotent_and_concurrency_is_bounded(harness):
    worker = harness["worker"](synchronous=False, max_workers=1)
    try:
        assert worker.start() is worker
        first_scheduler = worker._thread
        assert worker.start() is worker
        assert worker._thread is first_scheduler
    finally:
        worker.stop(timeout=2)


def test_runtime_limit_parks_job_between_steps_and_resume_continues(harness):
    def slow_read(_name, _args):
        time.sleep(1.05)
        return "first step result"

    harness["registry"]._outcomes["web_search"] = [slow_read]
    harness["rows"]["rows"] = [
        {"title": "First", "tool": "web_search"},
        {"title": "Second", "tool": "file_processor", "args": {"action": "info", "file_path": "second.txt", "save": False}},
    ]
    policy = autonomy.AutonomyPolicy(autonomy.AutonomyProfile(
        max_job_runtime_seconds=1, retry_backoff_seconds=0))
    worker = harness["worker"](synchronous=True, chosen_policy=policy)

    job = worker.submit("bounded job")

    assert job.status == bw.WAITING
    assert "runtime limit" in job.error.lower()
    assert engine.load(job.plan_id).steps[0].status == engine.DONE
    assert [name for name, _ in harness["registry"].calls] == ["web_search"]

    assert worker.resume(job.id)
    assert worker.status(job.id) == bw.COMPLETED
    assert [name for name, _ in harness["registry"].calls] == [
        "web_search", "file_processor"
    ]
