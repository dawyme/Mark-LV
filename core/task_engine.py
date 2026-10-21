"""
core/task_engine.py — multi-step goals, with a floor and a ceiling on both ends.

WHY THIS EXISTS
    Mark-LV answers a request with one tool call, and the Live model is told to
    chain a multi-step request itself. That works for two or three steps and
    falls apart past that: the model loses the thread, re-does a step it
    already did, or announces step four of a plan it never finished. There is
    also no record — start the app again and yesterday's half-finished job is
    gone, because the conversation it lived in is gone.

    This module gives a goal a spine: a plan is decomposed once, written to
    disk, and advanced one step at a time. The model stays in charge of what to
    say; the engine keeps track of what has been done.

WHAT IT DELIBERATELY DOES NOT DO
    It does not run anything itself. A step names a tool and its arguments;
    the caller runs it, through the same validated dispatch the rest of the app
    uses (core.action_loader), and reports the outcome back. That keeps the
    crash isolation, the signature introspection and the confirmation gate in
    one place instead of a second, weaker copy of them here.

    It does not loop. Every bound in this file is a number written down at the
    top, because an agent that decides for itself when to stop is an agent that
    eventually does not.

BOUNDS
    MAX_STEPS        how many steps a plan may have
    MAX_ATTEMPTS     how many times one step may be run
    MAX_PLANS_KEPT   finished-plan history to retain; unfinished plans are never evicted
    STEP_RESULT_CHARS  how much of a step's output is kept

SAFETY
    DEFAULT_ALLOWED_TOOLS names the built-in tools that may be considered by a
    task plan. core.autonomy applies the actual, configurable policy to each
    tool and its arguments: broad file/code actions are read-only only for an
    explicit safe action, while file-controller, computer-setting, and app
    launch actions retain their operation-aware gates. Unknown or sensitive
    operations require the same confirmation used elsewhere in Mark-LV.
    Decomposing a goal must not bypass a safeguard for a single action.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

# ── Bounds ────────────────────────────────────────────────────────────────────

MAX_STEPS         = 6      # a plan longer than this is a project, not a task
MAX_ATTEMPTS      = 2      # one retry: enough to recover, too few to spin
MAX_PLANS_KEPT    = 20
STEP_RESULT_CHARS = 400
GOAL_MAX_CHARS    = 300

# Tool candidates historically treated as routine by the task engine. The
# centralized autonomy profile further checks each tool's action arguments.
DEFAULT_ALLOWED_TOOLS = frozenset({
    "web_search",       # grounded search; returns text
    "research",         # the bounded research brief in actions/research.py
    "flight_finder",    # opens the browser; core.autonomy requires confirmation
    "file_controller",  # actual operation is checked by core.autonomy
    "file_processor",   # reads and summarises a local file
    "computer_settings", # only explicitly routine controls are autonomous
    "open_app",         # known local app aliases remain routine
    "code_helper",      # explains / reviews code, changes nothing
    "think",            # engine-internal: ask the model, touch nothing
})

# Everything else goes through core/confirm.py. Named here only so the refusal
# can say why; the decision is "not in the allow-list", never a blocklist that
# a new action would silently fall outside of.
_EXAMPLES_NEEDING_CONFIRMATION = (
    "file_controller", "computer_settings", "computer_control", "desktop",
    "send_message", "open_app", "dev_agent", "browser_control",
)


# ── Storage ───────────────────────────────────────────────────────────────────

def _base_dir() -> Path:
    import sys
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


STORE_PATH = _base_dir() / "memory" / "tasks.json"
_lock      = threading.RLock()
_ACTIVE_STEP_CLAIMS: set[tuple[str, int]] = set()


def set_store_path(path) -> None:
    """Point the store somewhere else. Used by the tests."""
    global STORE_PATH
    STORE_PATH = Path(path)


# ── Shapes ────────────────────────────────────────────────────────────────────

PENDING = "pending"
RUNNING = "running"
DONE    = "done"
FAILED  = "failed"
SKIPPED = "skipped"

PLANNED     = "planned"
IN_PROGRESS = "in_progress"
COMPLETED   = "completed"
ABANDONED   = "abandoned"
CANCELLED   = "cancelled"


@dataclass
class TaskStep:
    title:    str
    tool:     str
    args:     dict = field(default_factory=dict)
    status:   str = PENDING
    attempts: int = 0
    result:   str = ""
    error:    str = ""
    # Recovery of an interrupted non-idempotent action requires an explicit
    # review even if a later policy change would otherwise allow that tool.
    review_required: bool = False

    def needs_confirmation(self) -> bool:
        # Recovery review is an additional invariant, not a policy override.
        if self.review_required:
            return True
        # Keep action-level risk decisions centralized; fail closed for broad
        # tools if policy loading itself is unavailable.
        try:
            from core.autonomy import AutonomyPolicy
            return AutonomyPolicy.load().requires_confirmation(self.tool, self.args)[0]
        except Exception:
            if self.tool in {
                "file_controller", "file_processor", "computer_settings",
                "open_app", "code_helper", "flight_finder",
            }:
                return True
            return self.tool not in DEFAULT_ALLOWED_TOOLS


@dataclass
class TaskPlan:
    id:       str
    goal:     str
    created:  str
    status:   str = PLANNED
    steps:    list = field(default_factory=list)
    updated:  str = ""
    # Per-plan limits are persisted so policy changes or a restart do not
    # silently change the retry budget midway through an existing job.
    max_attempts: int = MAX_ATTEMPTS
    retry_backoff_seconds: float = 0.0
    # Failed/superseded steps remain auditable when a bounded re-plan replaces
    # the unfinished suffix of this plan.
    history: list = field(default_factory=list)

    # -- progress --
    def next_pending(self) -> Optional[TaskStep]:
        """The step to run next.

        A FAILED step counts only while it still has attempts left: once it is
        out, the plan is ABANDONED and pointing at that step again would just
        fail the same way a third time.
        """
        for step in self.steps:
            if step.status == PENDING:
                return step
            if step.status == FAILED and (
                    step.review_required
                    or step.attempts < max(1, int(self.max_attempts or MAX_ATTEMPTS))):
                return step
        return None

    def index_of(self, step: TaskStep) -> int:
        try:
            return self.steps.index(step)
        except ValueError:
            return -1

    def done_count(self) -> int:
        return sum(1 for s in self.steps if s.status in (DONE, SKIPPED))

    def is_finished(self) -> bool:
        return self.status in (COMPLETED, ABANDONED, CANCELLED)

    def summary(self) -> str:
        """One compact progress line — what the assistant says out loud."""
        if not self.steps:
            return f"{self.goal} — no steps."
        done = self.done_count()
        total = len(self.steps)
        if self.status == COMPLETED:
            return f"Done: {self.goal} ({done}/{total} steps)."
        current = self.next_pending()
        where = (f"next up: {current.title}" if current
                 else f"stopped at step {done + 1}")
        return f"{self.goal} — {done}/{total} steps done, {where}."


# ── Persistence ───────────────────────────────────────────────────────────────

def _read_raw() -> dict:
    with _lock:
        if not STORE_PATH.exists():
            return {"plans": []}
        try:
            data = json.loads(STORE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("plans"), list):
                return data
            raise ValueError("expected an object containing a plans array")
        except Exception as e:
            # Atomic writes prevent partial files; if a previous version or
            # external edit left corrupt JSON, preserve it for diagnosis rather
            # than overwriting potentially recoverable task history.
            backup = STORE_PATH.with_suffix(STORE_PATH.suffix + ".corrupt")
            if backup.exists():
                backup = STORE_PATH.with_name(
                    f"{STORE_PATH.name}.{int(time.time())}.{uuid.uuid4().hex[:6]}.corrupt")
            try:
                STORE_PATH.replace(backup)
                print(f"[Tasks] ⚠️ Corrupt state preserved as {backup.name}: {e}")
            except Exception:
                print(f"[Tasks] ⚠️ Could not read {STORE_PATH.name}: {e}")
            return {"plans": []}


def _write_raw(data: dict) -> None:
    plans = data.get("plans", [])
    if len(plans) > MAX_PLANS_KEPT:
        # History limits may trim finished plans, never unfinished work that a
        # background job still needs for recovery or cancellation.
        active = [row for row in plans if not isinstance(row, dict)
                  or row.get("status") not in (COMPLETED, ABANDONED, CANCELLED)]
        finished = [row for row in plans if isinstance(row, dict)
                    and row.get("status") in (COMPLETED, ABANDONED, CANCELLED)]
        keep_finished = max(0, MAX_PLANS_KEPT - len(active))
        data["plans"] = finished[-keep_finished:] + active if keep_finished else active
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=STORE_PATH.name + ".", dir=str(STORE_PATH.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, STORE_PATH)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def _plan_from_dict(raw: dict) -> Optional[TaskPlan]:
    try:
        steps = [TaskStep(title=str(s.get("title", "")).strip(),
                          tool=str(s.get("tool", "")).strip(),
                          args=s.get("args") if isinstance(s.get("args"), dict) else {},
                          status=s.get("status", PENDING),
                          attempts=int(s.get("attempts", 0) or 0),
                          result=str(s.get("result", "") or ""),
                          error=str(s.get("error", "") or ""),
                          review_required=bool(s.get("review_required", False)))
                 for s in (raw.get("steps") or []) if isinstance(s, dict)]
        steps = [s for s in steps if s.title and s.tool]
        plan = TaskPlan(id=str(raw.get("id", "")), goal=str(raw.get("goal", "")),
                        created=str(raw.get("created", "")),
                        status=raw.get("status", PLANNED), steps=steps,
                        updated=str(raw.get("updated", "")),
                        max_attempts=max(1, int(raw.get("max_attempts", MAX_ATTEMPTS) or MAX_ATTEMPTS)),
                        retry_backoff_seconds=max(0.0, float(raw.get("retry_backoff_seconds", 0.0) or 0.0)),
                        history=[h for h in (raw.get("history") or []) if isinstance(h, dict)][-20:])
        return plan if plan.id and plan.steps else None
    except Exception:
        return None


def _plan_to_dict(plan: TaskPlan) -> dict:
    out = asdict(plan)
    return out


def load_all() -> list[TaskPlan]:
    out = []
    for raw in _read_raw().get("plans", []):
        plan = _plan_from_dict(raw) if isinstance(raw, dict) else None
        if plan and plan.id:
            out.append(plan)
    return out


def load(plan_id: str) -> Optional[TaskPlan]:
    for plan in load_all():
        if plan.id == plan_id:
            return plan
    return None


def active() -> Optional[TaskPlan]:
    """The most recent unfinished plan — what 'carry on' refers to."""
    unfinished = [p for p in load_all() if not p.is_finished()]
    return unfinished[-1] if unfinished else None


def _save(plan: TaskPlan) -> None:
    with _lock:
        data = _read_raw()
        plan.updated = datetime.now().strftime("%Y-%m-%d %H:%M")
        rows = [p for p in data.get("plans", [])
                if not (isinstance(p, dict) and p.get("id") == plan.id)]
        rows.append(_plan_to_dict(plan))
        data["plans"] = rows
        _write_raw(data)


# ── Planning ──────────────────────────────────────────────────────────────────

_PLANNER_SYSTEM = (
    "You decompose one user goal into a short ordered list of steps for a "
    "desktop voice assistant. Reply with ONLY a JSON array. No prose, no "
    "markdown fence, no commentary."
)

_PLANNER_PROMPT = """Goal: {goal}

Return a JSON array of at most {max_steps} steps, in the order they must run:

[
  {{"title": "short imperative phrase", "tool": "<tool name>", "args": {{...}}}}
]

Tools you may use, and ONLY these:
{tools}

Rules:
1. Fewest steps that actually achieve the goal. One step if one is enough.
2. "think" means reason or write text yourself — use it for the final
   summary or any step that needs no tool.
3. Put every argument the tool needs in "args". Never invent an argument a
   tool does not take.
4. Never plan deletion, moving/renaming, overwriting, installation, code
   execution, or a message to another person. These are confirmation-gated.
5. `file_controller` may read/list/search/get info. Create a new file/folder or
   append only when the user explicitly asked; the live profile still gates all
   file writes. Never plan delete, move, rename, copy, or desktop organization.
6. `computer_settings` may only use explicit volume, brightness, or dark-mode
   adjustments. Restart, shutdown, WiFi, lock, close-app/window, typing, and
   other unrecognized controls require confirmation.
7. `open_app` is only for an explicitly requested, known local application alias;
   never pass a shell command or infer an app launch.
8. `file_processor` is broad: use only a named read-only action and include
   `"save": false`; never extract, convert, edit, run, or create files.
9. `code_helper` may only use `action: "explain"`. `weather_report` and
   `flight_finder` open a browser and require user confirmation; do not describe
   them as headless or autonomous. Use `web_search` for headless weather lookup.
10. If the goal cannot be reached with these tools, return [].

JSON:"""


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```[a-zA-Z]*\r?\n?", "", text)
    text = re.sub(r"\r?\n?```\s*$", "", text)
    return text.strip()


def parse_plan_json(raw: str, available_tools: set[str],
                    max_steps: int = MAX_STEPS) -> list[TaskStep]:
    """Turn a model's reply into validated steps. Never raises.

    A planner is the least reliable part of any agent, and the failure mode
    that matters is a step naming a tool that does not exist: it looks like
    progress and then fails three steps in. Unknown tools are dropped here,
    where there is still time to plan around them, rather than at execution.
    """
    text = _strip_fences(raw)
    if not text:
        return []
    if text.startswith("{"):            # some models wrap the array in an object
        try:
            inner = json.loads(text)
            if isinstance(inner, dict):
                for key in ("steps", "plan", "tasks"):
                    if isinstance(inner.get(key), list):
                        text = json.dumps(inner[key])
                        break
        except Exception:
            pass
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        rows = json.loads(text[start:end + 1])
    except Exception:
        return []
    if not isinstance(rows, list):
        return []

    allowed = set(available_tools or set()) | {"think"}
    steps: list[TaskStep] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title", "")).strip()
        tool  = str(row.get("tool", "")).strip().lower()
        args  = row.get("args")
        if not title or tool not in allowed:
            continue
        steps.append(TaskStep(title=title[:120], tool=tool,
                              args=args if isinstance(args, dict) else {}))
        if len(steps) >= max(1, min(int(max_steps or MAX_STEPS), MAX_STEPS)):
            break
    return steps


def build_plan(goal: str, available_tools: set[str],
               planner: Optional[Callable[[str, str], str]] = None,
               max_steps: int = MAX_STEPS, plan_id: str | None = None) -> TaskPlan:
    """Create and persist a plan for `goal`.

    `planner` is `fn(prompt, system) -> str`; when it is None the plan is
    decomposed through core.gemini, which walks its model ladder — and so
    reaches the FreeLLMAPI rung when the Gemini quota is gone. A plan is
    returned even when the model fails: one `think` step, so the assistant
    answers the goal directly instead of reporting that it could not plan.
    """
    goal = (goal or "").strip()[:GOAL_MAX_CHARS]
    now  = datetime.now().strftime("%Y-%m-%d %H:%M")
    chosen_id = str(plan_id or "").strip() or uuid.uuid4().hex[:10]
    plan = TaskPlan(id=chosen_id, goal=goal or "the request", created=now)

    if goal:
        prompt = _PLANNER_PROMPT.format(
            goal=goal,
            max_steps=max(1, min(max_steps, MAX_STEPS)),
            tools="\n".join(f"- {t}" for t in sorted(set(available_tools or set())
                                                     | {"think"})),
        )
        raw = ""
        try:
            if planner is not None:
                raw = planner(prompt, _PLANNER_SYSTEM) or ""
            else:
                from core import gemini
                raw = gemini.text(prompt, tier=gemini.SMART,
                                  config={"system_instruction": _PLANNER_SYSTEM},
                                  timeout_ms=45_000)
        except Exception as e:
            print(f"[Tasks] planner failed: {type(e).__name__}: {e}")
        plan.steps = parse_plan_json(raw, available_tools, max_steps=max_steps)

    if not plan.steps:
        plan.steps = [TaskStep(title="Answer the request directly", tool="think",
                               args={"question": goal})]
    _save(plan)
    return plan


_REPLAN_SYSTEM = (
    "You recover a task after one step failed. Return ONLY a JSON array of "
    "replacement steps that continue from the completed work. Never repeat a "
    "completed step. Do not plan destructive, external-facing, or executable "
    "actions. For file_processor use only an explicit read-only action and "
    'include "save": false. For code_helper use only action="explain". Do not '
    "treat weather_report or flight_finder as headless; both open a browser."
)


def replan_remaining(plan_id: str, failed_index: int, available_tools: set[str],
                     planner: Optional[Callable[[str, str], str]] = None,
                     max_steps: int = MAX_STEPS) -> Optional[TaskPlan]:
    """Replace a failed plan suffix once the caller decides a re-plan is due.

    Completed prefix steps are retained byte-for-byte. The failed and pending
    suffix is copied into ``plan.history`` before replacement, so attempts and
    errors remain available in the persisted report. Invalid/model-offline
    responses return None and never install a speculative fallback plan.
    """
    plan = load(plan_id)
    if (plan is None or failed_index < 0 or failed_index >= len(plan.steps)
            or plan.steps[failed_index].status != FAILED):
        return None
    completed = plan.steps[:failed_index]
    if any(s.status not in (DONE, SKIPPED) for s in completed):
        return None
    budget = max(0, min(MAX_STEPS, int(max_steps or MAX_STEPS)) - len(completed))
    if budget <= 0:
        return None

    suffix = plan.steps[failed_index:]
    # Do not send prior action output or arbitrary arguments back to the model:
    # a completed local file read can contain private text, even when the later
    # re-plan is only trying to choose an alternative next action.
    completed_notes = [
        {"title": s.title, "tool": s.tool, "status": "completed"}
        for s in completed if s.status == DONE
    ]
    pending_notes = [
        {"title": s.title, "tool": s.tool, "status": s.status}
        for s in suffix
    ]
    prompt = (
        f"Goal: {plan.goal}\n"
        f"Completed work (do not repeat): {json.dumps(completed_notes, ensure_ascii=False)}\n"
        f"Failed/unfinished plan suffix: {json.dumps(pending_notes, ensure_ascii=False)}\n"
        f"Create at most {budget} replacement step(s) using ONLY these registered tools:\n"
        + "\n".join(f"- {name}" for name in sorted(set(available_tools or set()) | {"think"}))
        + "\nRules: never repeat completed work; use file_processor only with an "
          'explicit read-only action and "save": false; use code_helper '
          'only with action="explain"; do not use weather_report or flight_finder '
          "or any write, executable, destructive, or external-facing action. Return [] "
          "if no safe, useful alternative exists. Each step must include title, "
          "tool, and args. JSON:"
    )
    try:
        if planner is not None:
            raw = planner(prompt, _REPLAN_SYSTEM) or ""
        else:
            from core import gemini
            raw = gemini.text(prompt, tier=gemini.SMART,
                              config={"system_instruction": _REPLAN_SYSTEM},
                              timeout_ms=30_000)
    except Exception as exc:
        print(f"[Tasks] re-plan failed: {type(exc).__name__}: {exc}")
        return None

    replacements = parse_plan_json(raw, available_tools, max_steps=budget)
    if not replacements:
        return None
    plan.history.append({
        "at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reason": "failed step; unfinished suffix replaced",
        "steps": [asdict(step) for step in suffix],
    })
    plan.history = plan.history[-20:]
    plan.steps = completed + replacements
    plan.status = IN_PROGRESS
    _save(plan)
    return load(plan_id)


# ── Execution ─────────────────────────────────────────────────────────────────

def start_step(plan_id: str, index: int) -> Optional[TaskStep]:
    """Atomically claim a step so parallel callers cannot execute it twice."""
    with _lock:
        plan = load(plan_id)
        if not plan or plan.is_finished() or index < 0 or index >= len(plan.steps):
            return None
        step = plan.steps[index]
        # A persisted success or in-flight attempt is final for this claim.
        if step.status in (DONE, SKIPPED, RUNNING):
            return None
        if (step.status == FAILED and not step.review_required
                and step.attempts >= max(1, int(plan.max_attempts or MAX_ATTEMPTS))):
            return None
        if step.status not in (PENDING, FAILED):
            return None
        if any(previous.status not in (DONE, SKIPPED) for previous in plan.steps[:index]):
            return None
        step.status = RUNNING
        step.attempts += 1
        if plan.status == PLANNED:
            plan.status = IN_PROGRESS
        _save(plan)
        _ACTIVE_STEP_CLAIMS.add((plan_id, index))
        return step


def active_step_indices(plan_id: str) -> set[int]:
    """Return step claims still executing in this process (not after restart)."""
    with _lock:
        return {index for claimed_plan, index in _ACTIVE_STEP_CLAIMS
                if claimed_plan == plan_id}


def load_with_active_steps(plan_id: str) -> tuple[Optional[TaskPlan], set[int]]:
    """Read the persisted plan and live claims under one execution lock."""
    with _lock:
        plan = load(plan_id)
        active = {index for claimed_plan, index in _ACTIVE_STEP_CLAIMS
                  if claimed_plan == plan_id}
        return plan, active


def record_result(plan_id: str, index: int, ok: bool, result: str = "",
                  error: str = "") -> Optional[TaskPlan]:
    """Persist a claimed step's outcome without reviving a cancelled plan."""
    with _lock:
        plan = load(plan_id)
        if not plan or index < 0 or index >= len(plan.steps):
            return None
        step = plan.steps[index]
        # Ignore stale/double callbacks. A cancellation may happen while the
        # action is running; still record its result, but keep the plan cancelled.
        claim = (plan_id, index)
        if step.status != RUNNING:
            _ACTIVE_STEP_CLAIMS.discard(claim)
            return plan
        was_cancelled = plan.status == CANCELLED
        if ok:
            step.status = DONE
            step.result = (result or "")[:STEP_RESULT_CHARS]
            step.error = ""
            step.review_required = False
        else:
            step.error = (error or result or "failed")[:STEP_RESULT_CHARS]
            step.status = FAILED

        if not was_cancelled:
            max_attempts = max(1, int(plan.max_attempts or MAX_ATTEMPTS))
            if all(s.status in (DONE, SKIPPED) for s in plan.steps):
                plan.status = COMPLETED
            elif step.status == FAILED and step.attempts >= max_attempts:
                # Out of attempts. A later step may depend on this one, so never
                # report a partial plan as successful.
                plan.status = ABANDONED
            else:
                plan.status = IN_PROGRESS
        _save(plan)
        _ACTIVE_STEP_CLAIMS.discard(claim)
        return plan


def update_step_args(plan_id: str, index: int, args: dict, *,
                     review_required: Optional[bool] = None,
                     error: Optional[str] = None) -> Optional[TaskPlan]:
    """Atomically persist repaired arguments before a retry or review gate."""
    if not isinstance(args, dict):
        return None
    with _lock:
        plan = load(plan_id)
        if not plan or plan.is_finished() or index < 0 or index >= len(plan.steps):
            return None
        step = plan.steps[index]
        if step.status not in (PENDING, FAILED):
            return None
        step.args = dict(args)
        if review_required is not None:
            step.review_required = bool(review_required)
        if error is not None:
            step.error = str(error)[:STEP_RESULT_CHARS]
        _save(plan)
        return plan


def skip_step(plan_id: str, index: int, reason: str = "") -> Optional[TaskPlan]:
    with _lock:
        plan = load(plan_id)
        if not plan or plan.is_finished() or index < 0 or index >= len(plan.steps):
            return None
        if plan.steps[index].status in (DONE, RUNNING):
            return None
        plan.steps[index].status = SKIPPED
        plan.steps[index].error = (reason or "skipped")[:STEP_RESULT_CHARS]
        plan.steps[index].review_required = False
        if all(s.status in (DONE, SKIPPED) for s in plan.steps):
            plan.status = COMPLETED
        _save(plan)
        return plan


def cancel(plan_id: str) -> bool:
    with _lock:
        plan = load(plan_id)
        if not plan or plan.is_finished():
            return False
        plan.status = CANCELLED
        _save(plan)
        return True


def can_retry(plan_id: str, index: int) -> bool:
    plan = load(plan_id)
    if not plan or index < 0 or index >= len(plan.steps):
        return False
    return plan.steps[index].attempts < max(1, int(plan.max_attempts or MAX_ATTEMPTS))


# ── Reporting ─────────────────────────────────────────────────────────────────

def render(plan: TaskPlan, max_steps: int = MAX_STEPS, policy=None) -> str:
    """The plan as the assistant reads it back — compact, no progress theatre.

    The result of each step is included only when it is short: a plan whose
    every step dumps a paragraph is a document, and this is a voice reply.
    """
    lines = [f"Goal: {plan.goal}", f"Status: {plan.status.replace('_', ' ')}",
             f"Steps ({plan.done_count()}/{len(plan.steps)} done):"]
    marks = {DONE: "✓", FAILED: "✗", SKIPPED: "–", RUNNING: "▶", PENDING: "·"}
    for i, step in enumerate(plan.steps[:max_steps], 1):
        mark = marks.get(step.status, "·")
        line = f" {i}. {mark} {step.title}"
        if step.status == DONE and step.result and len(step.result) <= 90:
            line += f" — {step.result}"
        if step.status == FAILED and step.error:
            line += f" — {step.error[:90]}"
        needs_confirmation = step.needs_confirmation()
        if policy is not None and not getattr(step, "review_required", False):
            try:
                needs_confirmation = policy.requires_confirmation(step.tool, step.args)[0]
            except Exception:
                pass
        if needs_confirmation:
            line += " [needs your confirmation]"
        lines.append(line)
    for event in plan.history[-3:]:
        old_steps = event.get("steps", []) if isinstance(event, dict) else []
        failed = next((s for s in old_steps
                       if isinstance(s, dict) and s.get("status") == FAILED), None)
        if failed:
            lines.append(" Replanned after: " + str(failed.get("title", "a failed step"))
                         + (f" — {str(failed.get('error', ''))[:90]}"
                            if failed.get("error") else ""))
    return "\n".join(lines)


def list_plans(limit: int = 5) -> str:
    plans = load_all()
    if not plans:
        return "No plans yet."
    lines = []
    for plan in plans[-limit:]:
        lines.append(f"- [{plan.id}] {plan.summary()}")
    return "\n".join(lines)
