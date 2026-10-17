"""
actions/task_agent.py — the voice interface to core/task_engine.py.

A self-describing action like every other in this folder: a TOOL dict plus a
handler, discovered at startup by core.action_loader, no edit to main.py.

WHAT IT ADDS TO THE CONVERSATION
    Mark-LV already chains two or three steps by itself. This is for the goal
    that does not fit in one breath: "find three flights, compare them and
    write the summary to a file". The plan is written down, run one step at a
    time, and survives a restart — so "carry on with that" means something the
    next morning.

HOW IT STAYS OUT OF THE WAY
    One step per call, and the plan's progress line is one sentence. The model
    still chooses its words; nothing here prints "Step 2 of 5 complete" in a
    voice, because an assistant that narrates its own bookkeeping is one you
    stop talking to.

WHAT IT WILL NOT DO UNASKED
    Deletion, moving/overwriting, sensitive system changes, unknown app launches,
    and external messages stay behind Mark-LV's existing on-screen confirmation
    gate. File writes also require the configured write permission. Routine
    local reads, known app aliases, and reversible volume/brightness controls
    retain their operation-aware defaults. The token comes from the interface,
    not the model, so decomposing a goal cannot bypass a single-action gate.
"""
from __future__ import annotations

import time

from core import confirm as confirm_gate
from core import task_engine as engine
from core import action_loader


# One step's work is given this long before the plan reports it as stuck rather
# than hanging the conversation. A search or a file read is a second; the ones
# that legitimately take longer (dev_agent) are the ones that need confirming,
# and those run behind the gate off the Qt thread.
STEP_BUDGET_SECONDS = 120.0
# `finish` will keep working for this long in total, then hand back the plan.
FINISH_BUDGET_SECONDS = 180.0


# ── Dispatch ──────────────────────────────────────────────────────────────────

def _available_tools() -> set[str]:
    """Tool names a plan step may name.

    Actions only, plus the engine's own `think`. Plugins are deliberately left
    out: they are downloaded one file at a time by the user, and a planner
    choosing one from a description is not a thing to do without the user
    watching.
    """
    registry = action_loader.get_registry()
    if not registry:
        return set()
    # These are orchestration/control tools, not goal-execution primitives.
    # Letting the planner select itself or the job manager creates nested jobs.
    return set(registry.names()) - {"task_agent", "jobs"}


def _think(question: str) -> str:
    """A step that needs no tool — reason, summarise, write.

    Goes through core.gemini, so it walks the model ladder and reaches the
    FreeLLMAPI rung when the Gemini text quota has run out.
    """
    from core import gemini
    text = gemini.text(question, tier=gemini.SMART, timeout_ms=60_000)
    if not text:
        raise RuntimeError(
            "The configured model provider returned no response. Check its "
            "credentials, network connection, and provider availability.")
    return text


def _run_tool(tool: str, args: dict, player=None, speak=None,
              confirmation_approved: bool = False) -> str:
    """Run one step through the SAME dispatch the rest of the app uses."""
    if tool == "think":
        return _think(str(args.get("question") or args.get("prompt") or ""))

    registry = action_loader.get_registry()
    if registry is None:
        return "No actions are loaded yet."
    if not registry.has(tool):
        return f"There is no tool called '{tool}'."
    ctx = {"player": player, "speak": speak, "response": None, "session_memory": None}
    if confirmation_approved and tool == "computer_settings":
        # This private context bit only comes from the UI callback below, never
        # from model-supplied tool arguments.
        ctx["_confirmation_approved"] = True
    # No `or "Done."` here: an empty result is a step that produced nothing,
    # and turning it into a word is how a plan reports success it did not earn.
    return registry.run(tool, dict(args or {}), ctx) or ""


# ── Recovery ──────────────────────────────────────────────────────────────────

_REPAIR_SYSTEM = (
    "You fix one step of a plan. Reply with ONLY a JSON object of corrected "
    "tool arguments, or {} if no change would help. No prose, no fence."
)


def _recovered_args(tool: str, step, error: str) -> dict:
    """Ask the model to repair a failed step's arguments. One call, bounded.

    Re-running the identical call is not error recovery, it is optimism. This
    hands the model the tool name, the arguments it sent and the error it got
    back, and takes a corrected argument set — or nothing, in which case the
    step is retried once as it was and then left alone.
    """
    try:
        import json as _json
        from core import gemini
        prompt = (
            f"Tool: {tool}\n"
            f"Arguments sent: {_json.dumps(step.args, ensure_ascii=False)}\n"
            f"Error returned: {str(error)[:400]}\n\n"
            "Return corrected arguments as a JSON object, or {} if the "
            "arguments were not the problem."
        )
        raw  = gemini.text(prompt, tier=gemini.FAST,
                           config={"system_instruction": _REPAIR_SYSTEM},
                           timeout_ms=20_000)
        data = _parse_json_object(raw)
        if isinstance(data, dict) and data:
            return data
    except Exception as e:
        print(f"[Tasks] recovery failed: {type(e).__name__}: {e}")
    return {}


def _parse_json_object(raw: str):
    import json as _json
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):] if "{" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return _json.loads(text[start:end + 1])
    except Exception:
        return None


# ── One step ──────────────────────────────────────────────────────────────────

def _log(player, text: str) -> None:
    if not player:
        return
    try:
        player.write_log(text)
    except Exception:
        pass


def _is_failed_result(text: str) -> bool:
    """Recognize explicit action failure strings without treating empty success
    fallbacks or a legitimate no-results search as completed work."""
    value = (text or "").strip()
    if not value:
        return True
    low = value.lower()
    if low in {"done", "done."}:
        return True
    if low.startswith(("tool '", "there is no tool called", "no actions are loaded",
                       "search failed:", "flight search failed", "error:", "failed:",
                       "failed to ", "could not ", "i could not ", "unable to ",
                       "permission denied:", "access denied:", "processing failed:",
                       "action failed", "request failed", "no application name provided",
                       "unsupported operating system", "file not found", "not found:",
                       "source not found:", "search path not found:", "path not found:",
                       "not a directory:", "not a file:", "no destination specified",
                       "no internet connection",
                       "sir, i couldn't open the browser", "sir, i couldn't reach gemini",
                       "please provide ", "please specify ", "please describe ",
                       "tell me what to look into", "no file path",
                       "path is not a file", "unsupported file type", "unknown file type",
                       "pillow is not installed", "pyautogui is not installed",
                       "python-docx not installed", "pandas not installed",
                       "pydub not installed", "ffmpeg not found", "no key specified",
                       "no text provided",
                       "python-pptx not installed", "interpreter not found", "execution error",
                       "invalid json", "info failed", "read failed", "stats failed",
                       "ai analysis failed", "ai image analysis failed", "ai processing failed",
                       "video transcription failed", "convert failed",
                       "filter failed", "sort failed", "list failed", "run failed",
                       "transcription failed", "extract audio failed", "unknown action:",
                       "unknown youtube action:", "unknown browser action:",
                       "unknown pdf action:", "unknown json action:", "unknown xml action:",
                       "unknown audio action:",
                       "unknown video action:", "unknown archive action:",
                       "unknown pptx action:", "unsupported archive format:",
                       "direct execution not supported", "no interpreter for ",
                       "execution timed out", "timed out after ",
                       "i couldn't find any flights", "i could not find anything")):
        return True
    if low.startswith("column ") and " not found" in low:
        return True
    if low.startswith("action '") and " is not available" in low:
        return True
    # A text-only model may legitimately answer with a limitation. Model
    # availability errors are raised by _think itself and handled above.
    return False


def _execute_step(plan_id: str, index: int, player=None, speak=None,
                  confirmation_approved: bool = False, should_stop=None) -> str:
    """Run one step, record it, and use the attempt budget if it fails.

    The budget is spent through engine.start_step(), which is the only place
    `attempts` is counted — so a step retried here and a step retried after a
    restart draw on the same allowance, and nothing can quietly get a third
    go because the counter lived in a local variable.

    Returns the sentence the assistant should say: short, and about the
    outcome rather than the machinery.
    """
    plan = engine.load(plan_id)
    if plan is None or index < 0 or index >= len(plan.steps):
        return "That plan is not there any more."
    # A confirmation can remain open while the user changes their mind. Never
    # let its callback execute a step after cancellation or another terminal
    # transition; start_step also checks this atomically against duplicate runs.
    if plan.is_finished():
        return f"That plan is already {plan.status}; the step was not run."
    step = plan.steps[index]

    _log(player, f"[Plan {plan_id}] {index + 1}/{len(plan.steps)} → {step.title}")

    attempt_args = dict(step.args or {})
    last_error   = ""

    def stop_requested() -> bool:
        if should_stop is None:
            return False
        try:
            return bool(should_stop())
        except Exception:
            # A failed cancellation check must not authorize another attempt.
            return True

    while True:
        if stop_requested():
            return "Stopped before starting another attempt."
        step = engine.start_step(plan_id, index)
        if step is None:
            return "That step is not there any more."

        try:
            result = _run_tool(step.tool, attempt_args, player=player, speak=speak,
                               confirmation_approved=confirmation_approved)
        except Exception as e:
            text, last_error = "", f"{type(e).__name__}: {e}"
        else:
            text, last_error = (result or "").strip(), ""

        # The registry catches a crashing handler and returns "Tool 'x' failed: …"
        # rather than raising, so a failure has to be recognised by its shape as
        # well as by an exception.
        failed = _is_failed_result(text)

        if not failed:
            engine.record_result(plan_id, index, ok=True, result=text)
            plan = engine.load(plan_id)
            if plan and plan.status == engine.COMPLETED:
                # A one-step plan is not a plan the user should hear reported;
                # it is the answer to what they asked.
                if len(plan.steps) == 1:
                    return text
                return f"That finished it. {plan.summary()}"
            return f"{step.title}: {text[:200]}" if text else f"{step.title}: done."

        last_error = last_error or text[:300]
        # Close out the current attempt before starting another one. Persisting
        # FAILED here lets start_step safely reject duplicate RUNNING attempts
        # while still allowing the configured retry budget.
        engine.record_result(plan_id, index, ok=False, error=last_error)
        current_plan = engine.load(plan_id)
        if (current_plan is not None and current_plan.is_finished()) or stop_requested():
            break

        # Confirmation applies to this exact operation and argument set. A
        # failed destructive/external action is never silently retried or
        # mutated under the approval just granted; a later attempt needs a new
        # confirmation from the user.
        if _needs_confirmation(step) or not engine.can_retry(plan_id, index):
            break
        fixed = _recovered_args(step.tool, step, last_error)
        if fixed:
            fixed = dict(fixed)
            needs_confirmation, confirmation_reason = (
                _autonomy_policy().requires_confirmation(step.tool, fixed)
            )
            if needs_confirmation:
                updated = engine.update_step_args(
                    plan_id, index, fixed, review_required=True,
                    error=f"Repaired arguments need explicit confirmation: {confirmation_reason}",
                )
                if updated is not None:
                    step = updated.steps[index]
                    last_error = ("Repaired arguments require explicit confirmation: "
                                  f"{confirmation_reason}")
                    _log(player, f"[Plan {plan_id}] repaired arguments parked for confirmation")
                break
            updated = engine.update_step_args(plan_id, index, fixed)
            if updated is None:
                break
            step = updated.steps[index]
            attempt_args = fixed
            _log(player, f"[Plan {plan_id}] retrying step {index + 1} with "
                         f"corrected arguments")
        current_plan = engine.load(plan_id)
        backoff = max(0.0, min(60.0, float(
            getattr(current_plan, "retry_backoff_seconds", 0.0) or 0.0)))
        if backoff:
            if should_stop is None:
                time.sleep(backoff)
            else:
                deadline = time.monotonic() + backoff
                while not stop_requested():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    time.sleep(min(0.1, remaining))
                if stop_requested():
                    break

    plan  = engine.load(plan_id)
    state = plan.status.replace("_", " ") if plan else "stopped"
    return (f"Step {index + 1} ({step.title}) did not work: {last_error[:160]}. "
            f"The plan is {state}.")


def _pending_confirmation(plan_id: str, index: int, step, player=None,
                          speak=None) -> str:
    """Park a sensitive step behind the on-screen gate.

    core.confirm.request() returns immediately with the sentence the model
    should say; the step itself runs only if the user presses CONFIRM, off the
    Qt thread. Nothing here blocks the conversation.
    """
    if confirm_gate.pending_title():
        return ("Something else is waiting for your confirmation on screen — "
                "answer that first.")

    detail = (f"{step.title}\n"
              f"tool: {step.tool}\n"
              f"arguments: {step.args or 'none'}")
    if getattr(step, "review_required", False):
        detail += ("\nWarning: the previous outcome is unknown. Re-running may "
                   "repeat an action that already took effect; review before approving.")

    approved_tool = step.tool
    approved_args = dict(step.args or {})

    def _do() -> str:
        current = engine.load(plan_id)
        if current is None or current.is_finished():
            return "That plan is no longer active; the step was not run."
        if index >= len(current.steps):
            return "That plan step changed; the step was not run. Review it and request confirmation again."
        current_step = current.steps[index]
        if current_step.tool != approved_tool or current_step.args != approved_args:
            return "That plan step changed; the step was not run. Review it and request confirmation again."
        return _execute_step(plan_id, index, player=player, speak=speak,
                             confirmation_approved=True)

    return confirm_gate.request(
        key=f"task-{plan_id}-{index}",
        title=f"Plan step {index + 1}: {step.tool}",
        detail=detail,
        run=_do,
    )


# ── Public actions ────────────────────────────────────────────────────────────

def _do_plan(goal: str, player=None, speak=None) -> str:
    if not goal:
        return "Tell me what you want done and I will plan it."

    # The app starts the durable worker during its normal lifecycle. Starting
    # lazily as well keeps the action usable in tests and alternate launchers.
    from core import autonomy
    from core import background_worker
    worker = background_worker.get_worker()
    policy = getattr(worker, "policy", None) or autonomy.AutonomyPolicy.load()
    if policy.profile.enable_background_jobs:
        worker.start()

    if not policy.profile.enable_background_jobs:
        # Preserve the original manual plan/run flow when the durable worker is
        # disabled. Planning and execution remain explicit in this mode.
        tools = _available_tools()
        plan = engine.build_plan(goal, tools,
                                 max_steps=policy.profile.max_plan_steps)
        plan.max_attempts = 1 + policy.profile.max_retries
        plan.retry_backoff_seconds = policy.profile.retry_backoff_seconds
        engine._save(plan)
        _log(player, f"[Plan {plan.id}] {goal} — {len(plan.steps)} step(s)")
        if len(plan.steps) == 1 and plan.steps[0].tool == "think":
            # Preserve the original direct-answer path when durable jobs are
            # disabled; there is no multi-step work to queue in that case.
            return _execute_step(plan.id, 0, player=player, speak=speak)
        return (f"Planned in {len(plan.steps)} steps ({plan.id}), but background "
                "jobs are disabled.\n" + engine.render(plan, policy=policy)
                + "\nYou can ask me to run or finish this plan.")

    # Plan creation itself can take a model round-trip. Submit the goal, not a
    # synchronously-built plan, so both planning and execution stay off the
    # voice/UI processing path. The worker reserves a durable plan ID first.
    try:
        job = worker.submit(goal)
    except Exception as e:
        return f"I could not queue that background goal: {e}. No work was started."
    if policy.profile.allow_autonomous:
        execution_note = "Eligible routine steps will run automatically."
    else:
        execution_note = (
            "Automatic execution is disabled by the autonomy profile; the job "
            "will wait for you to run or review its steps.")
    return (f"Started background job {job.id} to plan and execute this goal. "
            f"{execution_note}\n"
            "Use the jobs action to check progress, see the report, cancel, or "
            "resume it. Sensitive or external-facing steps wait for approval.")


def _next_step(plan_id: str) -> int:
    plan = engine.load(plan_id)
    if plan is None:
        return -1
    step = plan.next_pending()
    return plan.index_of(step) if step else -1


def _autonomy_policy():
    try:
        from core.background_worker import get_worker
        policy = getattr(get_worker(), "policy", None)
        if policy is not None:
            return policy
    except Exception:
        pass
    from core.autonomy import AutonomyPolicy
    return AutonomyPolicy.load()


def _needs_confirmation(step) -> bool:
    if getattr(step, "review_required", False):
        return True
    try:
        required, _reason = _autonomy_policy().requires_confirmation(step.tool, step.args)
        return required
    except Exception:
        return step.needs_confirmation()


def _job_for_plan(plan_id: str):
    try:
        from core import background_worker
        worker = background_worker.get_worker()
        return next((job for job in worker.list_jobs(100)
                     if job.plan_id == plan_id and not job.is_terminal()), None)
    except Exception:
        return None


def _do_run(plan_id: str, player=None, speak=None) -> str:
    plan = engine.load(plan_id) if plan_id else engine.active()
    if plan is None:
        return "There is no plan to carry on with."
    if plan.is_finished():
        return f"That plan is already {plan.status}."
    job = _job_for_plan(plan.id)
    if job is not None and job.status in ("queued", "running"):
        return f"Background job {job.id} is already running this plan. {job.summary()}"
    index = _next_step(plan.id)
    if index < 0:
        return plan.summary()
    step = plan.steps[index]
    if _needs_confirmation(step):
        return _pending_confirmation(plan.id, index, step, player=player, speak=speak)
    return _execute_step(plan.id, index, player=player, speak=speak)


def _do_finish(plan_id: str, player=None, speak=None) -> str:
    """Run the remaining steps, in order, until the plan ends or the budget does."""
    plan = engine.load(plan_id) if plan_id else engine.active()
    if plan is None:
        return "There is no plan to carry on with."
    if plan.is_finished():
        return f"That plan is already {plan.status}."
    job = _job_for_plan(plan.id)
    if job is not None and job.status in ("queued", "running"):
        return f"Background job {job.id} is already running this plan. {job.summary()}"

    started = time.monotonic()
    parked  = False
    while True:
        plan  = engine.load(plan.id)
        index = _next_step(plan.id)
        if index < 0:
            break
        if _needs_confirmation(plan.steps[index]):
            parked = True
            _pending_confirmation(plan.id, index, plan.steps[index],
                                  player=player, speak=speak)
            break
        _execute_step(plan.id, index, player=player, speak=speak)
        if time.monotonic() - started > FINISH_BUDGET_SECONDS:
            _log(player, f"[Plan {plan.id}] budget reached — handing back")
            break

    plan = engine.load(plan.id)
    if plan is None:
        return "The plan disappeared while it was running."
    if plan.status == engine.COMPLETED:
        return f"Done — {plan.goal}. All {len(plan.steps)} steps."
    if parked:
        return (f"{plan.summary()} One step needs your confirmation on screen "
                "before I can go on.")
    return plan.summary()


def _do_status(plan_id: str) -> str:
    plan = engine.load(plan_id) if plan_id else engine.active()
    if plan is None:
        return "No plan is running."
    try:
        from core.autonomy import AutonomyPolicy
        return engine.render(plan, policy=AutonomyPolicy.load())
    except Exception:
        return engine.render(plan)


def _do_cancel(plan_id: str) -> str:
    plan = engine.load(plan_id) if plan_id else engine.active()
    if plan is None:
        return "There is no plan to cancel."

    job = _job_for_plan(plan.id)
    active_step = bool(engine.active_step_indices(plan.id))
    if not plan.is_finished():
        engine.cancel(plan.id)
    current = engine.load(plan.id) or plan
    if current.status == engine.COMPLETED:
        return f"That plan already completed: {current.goal}."

    if job is not None:
        try:
            from core import background_worker
            worker = background_worker.get_worker()
            worker.cancel(job.id)
            latest_job = worker.get_job(job.id)
        except Exception:
            latest_job = None
        if latest_job is not None and latest_job.status == "cancelled":
            return f"Cancelled: {current.goal}."
        if latest_job is not None and latest_job.cancel_requested:
            return (f"Cancellation requested for {current.goal}; the active action "
                    "will finish before more work is stopped.")
        if latest_job is not None and latest_job.is_terminal():
            return f"That background job is already {latest_job.status}: {current.goal}."

    if current.status == engine.CANCELLED:
        if active_step:
            return (f"Cancellation recorded for {current.goal}; the active action "
                    "may finish, but no further step will start.")
        return f"Cancelled: {current.goal}."
    if current.is_finished():
        return f"That plan is already {current.status}: {current.goal}."
    return f"Could not confirm cancellation for {current.goal}; check its status."


def task_agent(parameters: dict, player=None, speak=None, session_memory=None) -> str:
    params  = parameters or {}
    action  = str(params.get("action", "")).strip().lower()
    goal    = str(params.get("goal", "") or "").strip()
    plan_id = str(params.get("plan_id", "") or "").strip()

    try:
        if action in ("plan", "new", "create"):
            return _do_plan(goal, player=player, speak=speak)
        if action == "run":
            return _do_run(plan_id, player=player, speak=speak)
        if action in ("finish", "complete", "run_all"):
            return _do_finish(plan_id, player=player, speak=speak)
        if action in ("status", "progress", "show"):
            return _do_status(plan_id)
        if action in ("list",):
            return engine.list_plans()
        if action in ("cancel", "stop", "abandon"):
            return _do_cancel(plan_id)
        return ("Say which of these you want: plan, run, finish, status, list "
                "or cancel.")
    except Exception as e:
        return f"The plan did not run: {e}"


# ── Tool declaration (auto-discovered by core/action_loader.py) ──────────────
TOOL = {
    "name": "task_agent",
    "description": (
        "Plans a multi-step goal using actual registered actions and automatically "
        "queues routine steps for durable background execution. Use it when a "
        "request needs several tool calls, or to inspect/continue an existing "
        "plan. Completed progress persists across restarts. Actions: 'plan' "
        "(needs goal) creates and starts a background job; 'run' takes the next "
        "step; 'finish' runs the remaining steps; 'status' reports progress; "
        "'list' shows recent plans; 'cancel' abandons one. Steps outside the "
        "autonomy allow-list wait for explicit confirmation/approval; never claim "
        "a step is done before its result is verified."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {
                "type": "STRING",
                "description": "plan | run | finish | status | list | cancel",
            },
            "goal": {
                "type": "STRING",
                "description": "What the user wants done, in their words (for action=plan)",
            },
            "plan_id": {
                "type": "STRING",
                "description": "A specific plan; omit to use the most recent unfinished one",
            },
        },
        "required": ["action"],
    },
    "handler": task_agent,
}
