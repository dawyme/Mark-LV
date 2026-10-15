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
    A step that writes, deletes, moves, installs, changes a setting, launches a
    program or sends anything to a person is parked behind the same on-screen
    confirmation the rest of Mark-LV uses (core/confirm.py). The token comes
    from the interface, not from the model, so decomposing a goal into steps
    cannot become a way around a gate that exists for single actions.
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
    return set(registry.names()) if registry else set()


def _think(question: str) -> str:
    """A step that needs no tool — reason, summarise, write.

    Goes through core.gemini, so it walks the model ladder and reaches the
    FreeLLMAPI rung when the Gemini text quota has run out.
    """
    from core import gemini
    text = gemini.text(question, tier=gemini.SMART, timeout_ms=60_000)
    return text or "I could not reason that one out."


def _run_tool(tool: str, args: dict, player=None, speak=None) -> str:
    """Run one step through the SAME dispatch the rest of the app uses."""
    if tool == "think":
        return _think(str(args.get("question") or args.get("prompt") or ""))

    registry = action_loader.get_registry()
    if registry is None:
        return "No actions are loaded yet."
    if not registry.has(tool):
        return f"There is no tool called '{tool}'."
    ctx = {"player": player, "speak": speak, "response": None, "session_memory": None}
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


def _execute_step(plan_id: str, index: int, player=None, speak=None) -> str:
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
    step = plan.steps[index]

    _log(player, f"[Plan {plan_id}] {index + 1}/{len(plan.steps)} → {step.title}")

    attempt_args = dict(step.args or {})
    last_error   = ""

    while True:
        step = engine.start_step(plan_id, index)
        if step is None:
            return "That step is not there any more."

        try:
            result = _run_tool(step.tool, attempt_args, player=player, speak=speak)
        except Exception as e:
            text, last_error = "", f"{type(e).__name__}: {e}"
        else:
            text, last_error = (result or "").strip(), ""

        # The registry catches a crashing handler and returns "Tool 'x' failed: …"
        # rather than raising, so a failure has to be recognised by its shape as
        # well as by an exception.
        failed = (not text
                  or (text.startswith("Tool '") and "failed:" in text)
                  or text.startswith("There is no tool called"))

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

        if not engine.can_retry(plan_id, index):
            break
        fixed = _recovered_args(step.tool, step, last_error)
        if fixed:
            attempt_args = fixed
            _log(player, f"[Plan {plan_id}] retrying step {index + 1} with "
                         f"corrected arguments")

    engine.record_result(plan_id, index, ok=False, error=last_error)
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

    def _do() -> str:
        return _execute_step(plan_id, index, player=player, speak=speak)

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
    tools = _available_tools()
    plan  = engine.build_plan(goal, tools)
    _log(player, f"[Plan {plan.id}] {goal} — {len(plan.steps)} step(s)")
    if len(plan.steps) == 1 and plan.steps[0].tool == "think":
        # The planner had nothing to decompose: answer the goal in one step,
        # which is what a direct tool call would have done anyway.
        return _execute_step(plan.id, 0, player=player, speak=speak)
    return (f"Planned in {len(plan.steps)} steps ({plan.id}).\n"
            + engine.render(plan)
            + "\nSay 'run' to take the first step, or ask me to finish the whole "
              "thing and I will run them in order.")


def _next_step(plan_id: str) -> int:
    plan = engine.load(plan_id)
    if plan is None:
        return -1
    step = plan.next_pending()
    return plan.index_of(step) if step else -1


def _do_run(plan_id: str, player=None, speak=None) -> str:
    plan = engine.load(plan_id) if plan_id else engine.active()
    if plan is None:
        return "There is no plan to carry on with."
    if plan.is_finished():
        return f"That plan is already {plan.status}."
    index = _next_step(plan.id)
    if index < 0:
        return plan.summary()
    step = plan.steps[index]
    if step.needs_confirmation():
        return _pending_confirmation(plan.id, index, step, player=player, speak=speak)
    return _execute_step(plan.id, index, player=player, speak=speak)


def _do_finish(plan_id: str, player=None, speak=None) -> str:
    """Run the remaining steps, in order, until the plan ends or the budget does."""
    plan = engine.load(plan_id) if plan_id else engine.active()
    if plan is None:
        return "There is no plan to carry on with."
    if plan.is_finished():
        return f"That plan is already {plan.status}."

    started = time.monotonic()
    parked  = False
    while True:
        plan  = engine.load(plan.id)
        index = _next_step(plan.id)
        if index < 0:
            break
        if plan.steps[index].needs_confirmation():
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
    return engine.render(plan)


def _do_cancel(plan_id: str) -> str:
    plan = engine.load(plan_id) if plan_id else engine.active()
    if plan is None:
        return "There is no plan to cancel."
    engine.cancel(plan.id)
    return f"Cancelled: {plan.goal}."


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
        "Plans and runs a multi-step goal, one step at a time, and remembers "
        "where it got to — including across restarts. Use it when a request "
        "needs more than two or three tool calls, or when the user asks to "
        "carry on with something already in progress. Prefer a single direct "
        "tool whenever one can do the job. Actions: 'plan' (needs goal) breaks "
        "the goal into steps; 'run' takes the next step; 'finish' runs the "
        "remaining steps in order; 'status' reports progress; 'list' shows "
        "recent plans; 'cancel' abandons one. A step that writes, deletes, "
        "changes a setting, launches a program or sends a message will ask the "
        "user to confirm on screen before it happens — say so, and never claim "
        "it is done."
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
