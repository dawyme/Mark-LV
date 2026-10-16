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
    MAX_PLANS_KEPT   how many finished plans stay on disk
    STEP_RESULT_CHARS  how much of a step's output is kept

SAFETY
    A step may only run a tool on DEFAULT_ALLOWED_TOOLS without asking. That
    list is read-only and local: searching, reading a file, explaining code.
    Consequential actions — such as deleting or overwriting files, changing
    system settings, sending external messages, or making purchases — need the
    same on-screen confirmation the rest of Mark-LV already uses (core/confirm.py) —
    a token issued by the interface, never by the model. Decomposing a goal
    into steps must not become a way around a safeguard that exists for single
    actions; if anything it needs it more, because a plan can reach a
    destructive tool three steps after the user agreed to something harmless.
"""
from __future__ import annotations

import json
import re
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

# Routine actions that may run without asking. Operations whose consequence
# varies by action (file management and computer settings) are classified in
# TaskStep.needs_confirmation() rather than gating every call to that tool.
DEFAULT_ALLOWED_TOOLS = frozenset({
    "web_search",       # grounded search; returns text
    "research",         # the bounded research brief in actions/research.py
    "weather_report",   # reads a forecast
    "flight_finder",    # reads a flight page
    "open_app",         # launches a requested local application
    "file_processor",   # reads and summarises a local file
    "code_helper",      # explains / reviews code, changes nothing
    "think",            # engine-internal: ask the model, touch nothing
})

# Everything else goes through core/confirm.py. Named here only so the refusal
# can say why; the decision is "not in the allow-list", never a blocklist that
# a new action would silently fall outside of.
_EXAMPLES_NEEDING_CONFIRMATION = (
    "file_controller", "computer_settings", "computer_control", "desktop",
    "send_message", "dev_agent", "browser_control",
)


# ── Storage ───────────────────────────────────────────────────────────────────

def _base_dir() -> Path:
    import sys
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


STORE_PATH = _base_dir() / "memory" / "tasks.json"
_lock      = threading.Lock()


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

    def needs_confirmation(self) -> bool:
        """Gate consequential operations, not whole categories of routine work.

        The same tool can be harmless or risky depending on the requested
        operation. Reading/listing files and ordinary computer controls should
        stay autonomous; destructive file operations and high-impact system
        controls still require the UI-issued confirmation token.
        """
        tool = str(self.tool or "").strip().lower()
        args = self.args if isinstance(self.args, dict) else {}
        action = str(args.get("action", "") or "").strip().lower()

        if tool == "file_controller":
            if action == "write":
                # Appending is non-destructive; replacing existing content is not.
                return not bool(args.get("append", False))
            return action in {"delete", "move", "rename", "organize_desktop"}

        if tool == "computer_settings":
            return action in {
                "restart", "shutdown", "toggle_wifi", "lock_screen",
                "close_app", "close_window",
            }

        return tool not in DEFAULT_ALLOWED_TOOLS


@dataclass
class TaskPlan:
    id:       str
    goal:     str
    created:  str
    status:   str = PLANNED
    steps:    list = field(default_factory=list)
    updated:  str = ""

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
            if step.status == FAILED and step.attempts < MAX_ATTEMPTS:
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
    if not STORE_PATH.exists():
        return {"plans": []}
    with _lock:
        try:
            data = json.loads(STORE_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) and isinstance(data.get("plans"), list) \
                else {"plans": []}
        except Exception as e:
            print(f"[Tasks] ⚠️ Could not read {STORE_PATH.name}: {e}")
            return {"plans": []}


def _write_raw(data: dict) -> None:
    plans = data.get("plans", [])
    if len(plans) > MAX_PLANS_KEPT:
        data["plans"] = plans[-MAX_PLANS_KEPT:]
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        STORE_PATH.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _plan_from_dict(raw: dict) -> Optional[TaskPlan]:
    try:
        steps = [TaskStep(title=str(s.get("title", "")).strip(),
                          tool=str(s.get("tool", "")).strip(),
                          args=s.get("args") if isinstance(s.get("args"), dict) else {},
                          status=s.get("status", PENDING),
                          attempts=int(s.get("attempts", 0) or 0),
                          result=str(s.get("result", "") or ""),
                          error=str(s.get("error", "") or ""))
                 for s in (raw.get("steps") or []) if isinstance(s, dict)]
        steps = [s for s in steps if s.title and s.tool]
        return TaskPlan(id=str(raw.get("id", "")), goal=str(raw.get("goal", "")),
                        created=str(raw.get("created", "")),
                        status=raw.get("status", PLANNED), steps=steps,
                        updated=str(raw.get("updated", "")))
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
4. Do not plan anything destructive. Nothing that deletes, moves, overwrites,
   installs, changes a system setting, or sends a message to a person.
5. If the goal cannot be reached with these tools, return [].

JSON:"""


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```[a-zA-Z]*\r?\n?", "", text)
    text = re.sub(r"\r?\n?```\s*$", "", text)
    return text.strip()


def parse_plan_json(raw: str, available_tools: set[str]) -> list[TaskStep]:
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
        if len(steps) >= MAX_STEPS:
            break
    return steps


def build_plan(goal: str, available_tools: set[str],
               planner: Optional[Callable[[str, str], str]] = None,
               max_steps: int = MAX_STEPS) -> TaskPlan:
    """Create and persist a plan for `goal`.

    `planner` is `fn(prompt, system) -> str`; when it is None the plan is
    decomposed through core.gemini, which walks its model ladder — and so
    reaches the FreeLLMAPI rung when the Gemini quota is gone. A plan is
    returned even when the model fails: one `think` step, so the assistant
    answers the goal directly instead of reporting that it could not plan.
    """
    goal = (goal or "").strip()[:GOAL_MAX_CHARS]
    now  = datetime.now().strftime("%Y-%m-%d %H:%M")
    plan = TaskPlan(id=uuid.uuid4().hex[:10], goal=goal or "the request",
                    created=now)

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
        plan.steps = parse_plan_json(raw, available_tools)

    if not plan.steps:
        plan.steps = [TaskStep(title="Answer the request directly", tool="think",
                               args={"question": goal})]
    _save(plan)
    return plan


# ── Execution ─────────────────────────────────────────────────────────────────

def start_step(plan_id: str, index: int) -> Optional[TaskStep]:
    """Mark a step running and persist that. Returns None if it is not there."""
    plan = load(plan_id)
    if not plan or index < 0 or index >= len(plan.steps):
        return None
    step = plan.steps[index]
    step.status   = RUNNING
    step.attempts += 1
    if plan.status == PLANNED:
        plan.status = IN_PROGRESS
    _save(plan)
    return step


def record_result(plan_id: str, index: int, ok: bool, result: str = "",
                  error: str = "") -> Optional[TaskPlan]:
    """Persist the outcome of a step, and the plan's status when it ends."""
    plan = load(plan_id)
    if not plan or index < 0 or index >= len(plan.steps):
        return None
    step = plan.steps[index]
    if ok:
        step.status = DONE
        step.result = (result or "")[:STEP_RESULT_CHARS]
        step.error  = ""
    else:
        step.error = (error or result or "failed")[:STEP_RESULT_CHARS]
        step.status = FAILED if step.attempts < MAX_ATTEMPTS else FAILED

    if all(s.status in (DONE, SKIPPED) for s in plan.steps):
        plan.status = COMPLETED
    elif step.status == FAILED and step.attempts >= MAX_ATTEMPTS:
        # Out of attempts. The plan stops here rather than limping on: a later
        # step usually depends on this one, and finishing "successfully" with a
        # hole in the middle is the kind of report nobody can act on.
        plan.status = ABANDONED
    else:
        plan.status = IN_PROGRESS
    _save(plan)
    return plan


def skip_step(plan_id: str, index: int, reason: str = "") -> Optional[TaskPlan]:
    plan = load(plan_id)
    if not plan or index < 0 or index >= len(plan.steps):
        return None
    plan.steps[index].status = SKIPPED
    plan.steps[index].error  = (reason or "skipped")[:STEP_RESULT_CHARS]
    if all(s.status in (DONE, SKIPPED) for s in plan.steps):
        plan.status = COMPLETED
    _save(plan)
    return plan


def cancel(plan_id: str) -> bool:
    plan = load(plan_id)
    if not plan:
        return False
    plan.status = CANCELLED
    _save(plan)
    return True


def can_retry(plan_id: str, index: int) -> bool:
    plan = load(plan_id)
    if not plan or index < 0 or index >= len(plan.steps):
        return False
    return plan.steps[index].attempts < MAX_ATTEMPTS


# ── Reporting ─────────────────────────────────────────────────────────────────

def render(plan: TaskPlan, max_steps: int = MAX_STEPS) -> str:
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
        if step.needs_confirmation():
            line += " [needs your confirmation]"
        lines.append(line)
    return "\n".join(lines)


def list_plans(limit: int = 5) -> str:
    plans = load_all()
    if not plans:
        return "No plans yet."
    lines = []
    for plan in plans[-limit:]:
        lines.append(f"- [{plan.id}] {plan.summary()}")
    return "\n".join(lines)
