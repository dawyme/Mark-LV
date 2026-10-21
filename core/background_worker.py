"""Persistent background jobs built on Mark-LV's existing task engine.

Jobs are planned with ``core.task_engine``, executed through the existing
``core.action_loader`` dispatch via ``actions.task_agent``, and persisted as
atomic JSON in ``memory/jobs.json``. The worker runs on daemon threads, not the
Qt UI or Live voice event loop. Only policy-approved routine steps are
autonomous; unknown, write, destructive, external-facing, and security-sensitive
actions wait for a human instead.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

MAX_JOBS_KEPT = 100
POLL_INTERVAL = 0.5
DEFAULT_MAX_WORKERS = 2
DEFAULT_MAX_RUNTIME_SECONDS = 300
DEFAULT_MAX_ATTEMPTS = 3

QUEUED = "queued"
RUNNING = "running"
WAITING = "waiting"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"
ALL_STATES = (QUEUED, RUNNING, WAITING, COMPLETED, FAILED, CANCELLED)
TERMINAL_STATES = (COMPLETED, FAILED, CANCELLED)


def _base_dir() -> Path:
    import sys
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


_DEFAULT_STORE = _base_dir() / "memory" / "jobs.json"
_store_path = _DEFAULT_STORE
_store_lock = threading.RLock()
_RUNNING_JOB_LOCK = threading.Lock()
_RUNNING_JOB_IDS: set[str] = set()


def set_store_path(path) -> None:
    global _store_path
    _store_path = Path(path)


def get_store_path() -> Path:
    return _store_path


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def _read_raw(path: Path) -> dict:
    with _store_lock:
        if not path.exists():
            return {"jobs": []}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("jobs"), list):
                return raw
            raise ValueError("expected an object containing a jobs array")
        except Exception as exc:
            backup = path.with_suffix(path.suffix + ".corrupt")
            if backup.exists():
                backup = path.with_name(
                    f"{path.name}.{int(time.time())}.{uuid.uuid4().hex[:6]}.corrupt")
            try:
                path.replace(backup)
                print(f"[Jobs] ⚠️ Corrupt state preserved as {backup.name}: {exc}")
            except Exception:
                print(f"[Jobs] ⚠️ Could not read {path.name}: {exc}")
            return {"jobs": []}


def _write_raw(path: Path, data: dict) -> None:
    data = dict(data)
    jobs = data.get("jobs", [])
    if len(jobs) > MAX_JOBS_KEPT:
        # Never evict active work merely to retain more terminal history.
        active = [row for row in jobs if not isinstance(row, dict)
                  or row.get("status") not in TERMINAL_STATES]
        terminal = [row for row in jobs if isinstance(row, dict)
                    and row.get("status") in TERMINAL_STATES]
        keep_terminal = max(0, MAX_JOBS_KEPT - len(active))
        data["jobs"] = terminal[-keep_terminal:] + active if keep_terminal else active
    with _store_lock:
        _atomic_write(path, data)


@dataclass
class Job:
    id: str
    goal: str
    plan_id: str = ""
    # Snapshot duplicates the engine's persisted plan so the job can be
    # reconstructed if tasks.json is damaged; task_engine remains canonical.
    plan_snapshot: dict = field(default_factory=dict)
    status: str = QUEUED
    created: str = ""
    updated: str = ""
    started: str = ""
    finished: str = ""
    attempts: int = 0                 # number of worker starts/resumes
    max_attempts: int = DEFAULT_MAX_ATTEMPTS  # compatibility / audit field
    priority: int = 0
    log: list[str] = field(default_factory=list)
    result: str = ""
    error: str = ""
    worker_id: str = ""
    cancel_requested: bool = False
    metadata: dict = field(default_factory=dict)

    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATES

    def summary(self) -> str:
        text = f"[{self.id}] {self.status}: {self.goal}"
        if self.result:
            text += f" — {self.result[:120]}"
        if self.error:
            text += f" — {self.error[:120]}"
        return text


def _job_from_dict(raw: dict) -> Optional[Job]:
    if not isinstance(raw, dict):
        return None
    try:
        job = Job(
            id=str(raw.get("id", "")).strip(),
            goal=str(raw.get("goal", "")).strip(),
            plan_id=str(raw.get("plan_id", "") or "").strip(),
            plan_snapshot=raw.get("plan_snapshot") if isinstance(raw.get("plan_snapshot"), dict) else {},
            status=str(raw.get("status", QUEUED) or QUEUED).strip().lower(),
            created=str(raw.get("created", "") or ""),
            updated=str(raw.get("updated", "") or ""),
            started=str(raw.get("started", "") or ""),
            finished=str(raw.get("finished", "") or ""),
            attempts=max(0, int(raw.get("attempts", 0) or 0)),
            max_attempts=max(1, int(raw.get("max_attempts", DEFAULT_MAX_ATTEMPTS) or DEFAULT_MAX_ATTEMPTS)),
            priority=int(raw.get("priority", 0) or 0),
            log=[str(x) for x in raw.get("log", [])][-100:] if isinstance(raw.get("log"), list) else [],
            result=str(raw.get("result", "") or ""),
            error=str(raw.get("error", "") or ""),
            worker_id=str(raw.get("worker_id", "") or ""),
            cancel_requested=bool(raw.get("cancel_requested", False)),
            metadata=raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {},
        )
        if not job.id:
            return None
        if job.status not in ALL_STATES:
            job.status = QUEUED
        return job
    except Exception:
        return None


def _load_all(path: Path) -> list[Job]:
    with _store_lock:
        rows = _read_raw(path).get("jobs", [])
        return [job for item in rows if (job := _job_from_dict(item)) is not None]


def _load(path: Path, job_id: str) -> Optional[Job]:
    return next((job for job in _load_all(path) if job.id == job_id), None)


def _save(path: Path, job: Job) -> None:
    job.updated = _now()
    job.log = list(job.log[-100:])
    with _store_lock:
        data = _read_raw(path)
        rows = [row for row in data.get("jobs", [])
                if not (isinstance(row, dict) and row.get("id") == job.id)]
        rows.append(asdict(job))
        data["jobs"] = rows
        _write_raw(path, data)


def _delete(path: Path, job_id: str) -> bool:
    with _store_lock:
        data = _read_raw(path)
        rows = data.get("jobs", [])
        remaining = [row for row in rows
                     if not (isinstance(row, dict) and row.get("id") == job_id)]
        if len(remaining) == len(rows):
            return False
        data["jobs"] = remaining
        _write_raw(path, data)
        return True


class JobStore:
    """Thread-safe, atomic JSON storage for jobs."""
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else get_store_path()

    def set_path(self, path) -> None:
        self.path = Path(path)

    def load_all(self) -> list[Job]:
        return _load_all(self.path)

    def load(self, job_id: str) -> Optional[Job]:
        return _load(self.path, job_id)

    def save(self, job: Job) -> None:
        _save(self.path, job)

    def delete(self, job_id: str) -> bool:
        return _delete(self.path, job_id)

    def clear(self) -> None:
        with _store_lock:
            _atomic_write(self.path, {"jobs": []})


def load_all() -> list[Job]:
    return _load_all(get_store_path())


def load(job_id: str) -> Optional[Job]:
    return _load(get_store_path(), job_id)


def save(job: Job) -> None:
    _save(get_store_path(), job)


def delete(job_id: str) -> bool:
    return _delete(get_store_path(), job_id)


def _snapshot(plan) -> dict:
    try:
        return asdict(plan)
    except Exception:
        return {}


class BackgroundWorker:
    """A single scheduler with a bounded set of daemon job threads.

    Multiple calls to ``start`` on one worker are idempotent. A process-wide
    active-job set protects against two worker instances dispatching the same
    persisted job concurrently.
    """
    def __init__(self, store_path=None, max_workers: Optional[int] = None,
                 poll_interval: float = POLL_INTERVAL, autonomy_policy=None,
                 synchronous: bool = False):
        self.store_path = Path(store_path) if store_path is not None else get_store_path()
        self.store = JobStore(self.store_path)
        if autonomy_policy is None:
            from core.autonomy import AutonomyPolicy
            autonomy_policy = AutonomyPolicy.load()
        self.policy = autonomy_policy
        policy_limit = max(1, int(getattr(self.policy, "max_concurrent_jobs",
                                           DEFAULT_MAX_WORKERS)))
        requested_workers = policy_limit if max_workers is None else int(max_workers)
        self.max_workers = max(1, min(16, policy_limit, requested_workers))
        self.poll_interval = max(0.1, min(10.0, float(poll_interval or POLL_INTERVAL)))
        self.synchronous = bool(synchronous)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._active: set[str] = set()
        self._threads: dict[str, threading.Thread] = {}
        self._active_lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._started = False

    def is_running(self) -> bool:
        with self._lifecycle_lock:
            if self.synchronous:
                return self._started
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> "BackgroundWorker":
        with self._lifecycle_lock:
            if not getattr(getattr(self.policy, "profile", None), "enable_background_jobs", True):
                return self
            if self.is_running():
                return self
            self._stop.clear()
            self.recover()
            self._started = True
            if not self.synchronous:
                self._thread = threading.Thread(target=self._scheduler_loop,
                                               name="marklv-job-scheduler", daemon=True)
                self._thread.start()
            return self

    def stop(self, timeout: float = 10.0) -> None:
        """Stop at a step boundary and persist unfinished work as queued.

        A call already inside an action is not forcibly killed. The bounded wait
        gives it a chance to finish; if it cannot, its durable RUNNING state is
        recovered on the next launch. Daemon worker threads never hold the UI
        process open indefinitely.
        """
        with self._lifecycle_lock:
            self._stop.set()
            deadline = time.monotonic() + max(0.0, timeout)
            if self._thread is not None:
                self._thread.join(max(0.0, deadline - time.monotonic()))
            with self._active_lock:
                threads = list(self._threads.values())
            for thread in threads:
                thread.join(max(0.0, deadline - time.monotonic()))
            self._started = False

    def _recover_interrupted_steps(self, plan):
        """Recover stale RUNNING steps, but leave this process's live claims alone."""
        from core import task_engine as engine

        latest, active = engine.load_with_active_steps(plan.id)
        if latest is not None:
            plan = latest
        running = [index for index, step in enumerate(plan.steps)
                   if step.status == engine.RUNNING]
        if not running:
            return "none", False, False
        if any(index in active for index in running):
            return "active", False, False

        max_attempts = max(1, int(plan.max_attempts or engine.MAX_ATTEMPTS))
        review_required = False
        exhausted = False
        for index in running:
            step = plan.steps[index]
            allowed, _ = self.policy.can_run_autonomously(step.tool, step.args)
            replayable = getattr(self.policy, "can_replay_after_crash", None)
            safe_to_replay = (allowed and
                              (replayable(step.tool, step.args) if callable(replayable)
                               else step.tool in getattr(engine, "DEFAULT_ALLOWED_TOOLS", ())))
            step.status = engine.FAILED
            if safe_to_replay and step.attempts < max_attempts:
                # Preserve the consumed attempt; only idempotent read-only
                # built-ins may be retried after an unknown outcome.
                step.error = "Interrupted before the result was recorded; safe retry queued."
            elif safe_to_replay:
                step.error = "Interrupted after the retry budget was exhausted."
                exhausted = True
            else:
                # Even if policy later changes, an uncertain outcome needs a
                # new explicit human approval before any replay.
                step.error = "Outcome unknown after interruption; explicit review required."
                step.review_required = True
                review_required = True

        if exhausted:
            plan.status = engine.ABANDONED
        else:
            plan.status = engine.IN_PROGRESS
        engine._save(plan)
        return "recovered", review_required, exhausted

    def recover(self) -> list[str]:
        """Requeue interrupted eligible jobs and adopt unfinished engine plans."""
        recovered: list[str] = []
        with _RUNNING_JOB_LOCK:
            currently_running = set(_RUNNING_JOB_IDS)
        jobs = self.store.load_all()
        from core import task_engine as engine

        for job in jobs:
            if job.is_terminal() or job.id in currently_running:
                continue

            plan = self._load_or_restore_plan(job) if (job.plan_id or job.plan_snapshot) else None

            # A crash can occur after the canonical plan is finalized but
            # before the job row records its terminal state, including a job
            # that was waiting for manual approval. Trust the persisted plan.
            if plan is not None and plan.is_finished():
                job.plan_snapshot = _snapshot(plan)
                if plan.status == engine.COMPLETED:
                    job.status = COMPLETED
                    job.result = plan.summary()
                    job.error = ""
                elif plan.status == engine.CANCELLED:
                    job.status = CANCELLED
                else:
                    job.status = FAILED
                    job.error = job.error or plan.summary()
                job.finished = job.finished or _now()
                job.log.append("Finalized from the persisted task plan during recovery.")
                self.store.save(job)
                continue

            # A durable cancellation request wins over recovery. Never replay
            # an in-flight operation after the user asked to stop.
            if job.cancel_requested:
                if plan is not None:
                    engine.cancel(plan.id)
                    plan = engine.load(plan.id) or plan
                    job.plan_snapshot = _snapshot(plan)
                job.status = CANCELLED
                job.error = "Cancelled before recovery; no further work was run."
                job.finished = job.finished or _now()
                job.log.append("Cancellation request honored during recovery.")
                self.store.save(job)
                continue

            recovery_state, review_required, exhausted = (
                self._recover_interrupted_steps(plan) if plan is not None
                else ("none", False, False)
            )
            if plan is not None:
                plan = engine.load(plan.id) or plan
            if recovery_state == "active":
                # This can be a user-confirmed task_agent step running outside
                # the scheduler. Do not mistake it for a crash or replay it.
                continue

            if recovery_state == "recovered":
                job.plan_snapshot = _snapshot(plan)
                if exhausted or plan.status == engine.ABANDONED:
                    job.status = FAILED
                    job.error = "Interrupted step exhausted its retry budget."
                    job.finished = _now()
                elif review_required:
                    job.status = WAITING
                    job.error = "Interrupted action outcome needs explicit human review."
                    job.log.append("Recovery paused because automatic replay is not safe.")
                else:
                    job.status = QUEUED
                    job.error = ""
                    job.cancel_requested = False
                    job.log.append("Recovered after interruption; safe unfinished work will resume.")
                    recovered.append(job.id)
                self.store.save(job)
                continue

            # RUNNING with no in-flight step means the crash happened between
            # steps (or while a plan was being created). QUEUED jobs are already
            # recoverable; WAITING jobs stay parked until the user resumes.
            if job.status == RUNNING:
                can_resume = (plan is not None or bool(job.metadata.get("_plan_pending"))
                              or not job.plan_id)
                if plan is not None:
                    job.plan_snapshot = _snapshot(engine.load(plan.id) or plan)
                if can_resume:
                    job.status = QUEUED
                    job.error = ""
                    job.cancel_requested = False
                    job.log.append("Recovered between steps; unfinished work will resume.")
                    recovered.append(job.id)
                else:
                    job.status = WAITING
                    job.error = "Interrupted job has no recoverable plan; review required."
                    job.log.append(job.error)
                self.store.save(job)
            elif job.status == QUEUED:
                recovered.append(job.id)

        # Adopt unfinished legacy plans created before the job worker existed.
        try:
            known = {j.plan_id for j in self.store.load_all() if j.plan_id}
            for plan in engine.load_all():
                if plan.is_finished() or plan.id in known:
                    continue
                recovery_state, review_required, exhausted = (
                    self._recover_interrupted_steps(plan)
                )
                plan = engine.load(plan.id) or plan
                if recovery_state == "active":
                    continue
                if exhausted or plan.status == engine.ABANDONED:
                    status = FAILED
                    error = "Interrupted step exhausted its retry budget."
                elif review_required:
                    status = WAITING
                    error = "Interrupted action outcome needs explicit human review."
                else:
                    status = QUEUED
                    error = ""
                job = Job(id=uuid.uuid4().hex[:10], goal=plan.goal or "resumed plan",
                          plan_id=plan.id, plan_snapshot=_snapshot(plan), status=status,
                          created=_now(), error=error,
                          log=["Recovered unfinished task-engine plan."])
                if status == QUEUED:
                    recovered.append(job.id)
                self.store.save(job)
                known.add(plan.id)
        except Exception as exc:
            print(f"[Jobs] Unfinished-plan recovery skipped: {exc}")
        return recovered

    def _load_or_restore_plan(self, job: Job):
        from core import task_engine as engine
        plan = engine.load(job.plan_id) if job.plan_id else None
        if plan is not None:
            return plan
        if job.plan_snapshot:
            try:
                plan = engine._plan_from_dict(job.plan_snapshot)
                if plan is not None:
                    engine._save(plan)
                    job.plan_id = plan.id
                    return plan
            except Exception as exc:
                job.error = f"Could not restore plan snapshot: {exc}"
        return None

    def submit(self, goal: str, priority: int = 0, metadata: dict | None = None,
               plan_id: str = "") -> Job:
        goal = (goal or "").strip()
        if not goal:
            raise ValueError("goal is required")

        # Reserve the plan ID before persisting the job. If the process stops
        # while the planner is working, recovery can rebuild into that same ID
        # instead of creating a second orphan plan/job for the same submission.
        with _store_lock:
            jobs = self.store.load_all()
            if plan_id:
                duplicate = next((j for j in jobs if j.plan_id == plan_id), None)
                if duplicate is not None:
                    return duplicate
                from core import task_engine as engine
                plan = engine.load(plan_id)
                if plan is None:
                    raise ValueError(f"task plan '{plan_id}' does not exist")
            else:
                plan = None
                plan_id = uuid.uuid4().hex[:10]

            active_count = sum(j.status in (QUEUED, RUNNING, WAITING) for j in jobs)
            ok, reason = self.policy.check_submit(active_count)
            if not ok:
                raise RuntimeError(reason)
            job_metadata = dict(metadata or {})
            job_metadata.pop("_plan_pending", None)
            if plan is None:
                job_metadata["_plan_pending"] = True
            job = Job(
                id=uuid.uuid4().hex[:10], goal=goal, plan_id=plan_id,
                plan_snapshot=_snapshot(plan) if plan is not None else {},
                status=QUEUED, created=_now(), priority=int(priority or 0),
                max_attempts=1 + int(getattr(self.policy, "max_retries", 1)),
                metadata=job_metadata,
            )
            self.store.save(job)
        if not self.is_running():
            self.start()
        if self.synchronous:
            self._run_job(job.id)
            return self.store.load(job.id) or job
        self._dispatch(job.id)
        return self.store.load(job.id) or job

    submit_job = submit
    enqueue = submit

    def _dispatch(self, job_id: str) -> None:
        with self._active_lock:
            if job_id in self._active or len(self._active) >= self.max_workers:
                return
            job = self.store.load(job_id)
            if job is None or job.status != QUEUED:
                return
            self._active.add(job_id)
            thread = threading.Thread(target=self._run_job, args=(job_id,),
                                      name=f"marklv-job-{job_id}", daemon=True)
            self._threads[job_id] = thread
            thread.start()

    def _scheduler_loop(self) -> None:
        while not self._stop.wait(self.poll_interval):
            try:
                jobs = [j for j in self.store.load_all() if j.status == QUEUED]
                jobs.sort(key=lambda j: (-j.priority, j.created))
                for job in jobs:
                    with self._active_lock:
                        if len(self._active) >= self.max_workers:
                            break
                    self._dispatch(job.id)
            except Exception as exc:
                print(f"[Jobs] scheduler error: {exc}")

    def _save_job_state(self, job: Job) -> None:
        # Merge a concurrent cancellation request rather than overwriting it
        # with the worker's last in-memory copy of the job.
        with _store_lock:
            latest = self.store.load(job.id)
            if latest is not None and latest.cancel_requested:
                job.cancel_requested = True
            self.store.save(job)

    def _persist_plan(self, job: Job, plan) -> None:
        job.plan_id = plan.id
        job.plan_snapshot = _snapshot(plan)
        self._save_job_state(job)

    def _should_stop_job(self, job_id: str) -> bool:
        if self._stop.is_set():
            return True
        latest = self.store.load(job_id)
        return latest is None or latest.cancel_requested or latest.status == CANCELLED

    def _run_job(self, job_id: str) -> None:
        # A second worker instance in the same process cannot execute this job.
        with _RUNNING_JOB_LOCK:
            if job_id in _RUNNING_JOB_IDS:
                with self._active_lock:
                    self._active.discard(job_id)
                    self._threads.pop(job_id, None)
                return
            _RUNNING_JOB_IDS.add(job_id)
        job = None
        try:
            with _store_lock:
                job = self.store.load(job_id)
                if job is None or job.status not in (QUEUED, RUNNING):
                    return
                if job.cancel_requested:
                    job.status = CANCELLED
                    job.finished = _now()
                    self.store.save(job)
                    return
                if self._stop.is_set():
                    job.status = QUEUED
                    self.store.save(job)
                    return

                job.status = RUNNING
                job.attempts += 1
                job.started = job.started or _now()
                job.worker_id = threading.current_thread().name
                job.log.append("Worker started.")
                self.store.save(job)

            from core import action_loader, task_engine as engine
            from actions import task_agent

            plan = self._load_or_restore_plan(job)
            plan_pending = bool(job.metadata.get("_plan_pending"))
            if plan is None and job.plan_id and not plan_pending:
                job.status = WAITING
                job.error = "Persisted plan and recovery snapshot are missing; review required."
                job.log.append(job.error)
                self._save_job_state(job)
                return
            if plan is None:
                if not job.plan_id:
                    job.plan_id = uuid.uuid4().hex[:10]
                    job.metadata["_plan_pending"] = True
                    self._save_job_state(job)
                registry = action_loader.get_registry()
                tools = (registry.names() - {"task_agent", "jobs"}
                         if registry is not None else set())
                plan = engine.build_plan(
                    job.goal, tools, max_steps=self.policy.profile.max_plan_steps,
                    plan_id=job.plan_id)
                plan.max_attempts = 1 + self.policy.profile.max_retries
                plan.retry_backoff_seconds = self.policy.profile.retry_backoff_seconds
                engine._save(plan)
                job.metadata.pop("_plan_pending", None)
                self._persist_plan(job, plan)
            else:
                # Fill limits missing from pre-profile plans only; limits that
                # were explicitly persisted with a job remain stable.
                if not getattr(plan, "max_attempts", 0):
                    plan.max_attempts = 1 + self.policy.profile.max_retries
                    plan.retry_backoff_seconds = self.policy.profile.retry_backoff_seconds
                    engine._save(plan)
                job.metadata.pop("_plan_pending", None)
                self._persist_plan(job, plan)

            run_started = time.monotonic()
            while True:
                latest = self.store.load(job_id)
                if latest is None:
                    return
                job.cancel_requested = latest.cancel_requested
                if job.cancel_requested:
                    current_plan = engine.load(job.plan_id) if job.plan_id else None
                    if current_plan is not None and current_plan.status == engine.COMPLETED:
                        job.plan_snapshot = _snapshot(current_plan)
                        job.status = COMPLETED
                        job.result = current_plan.summary()
                        job.error = ""
                        job.finished = _now()
                        job.log.append("Cancellation arrived after the final step completed.")
                        self._save_job_state(job)
                        return
                    if job.plan_id:
                        engine.cancel(job.plan_id)
                    job.status = CANCELLED
                    job.finished = _now()
                    job.log.append("Cancelled after the current action returned.")
                    self.store.save(job)
                    return
                if self._stop.is_set():
                    job.status = QUEUED
                    job.log.append("Paused at shutdown; will resume on next application start.")
                    self._persist_plan(job, engine.load(job.plan_id) or plan)
                    return

                plan = engine.load(job.plan_id)
                if plan is None:
                    plan = self._load_or_restore_plan(job)
                if plan is None:
                    raise RuntimeError("persisted plan is missing and could not be restored")
                if plan.is_finished():
                    break
                step = plan.next_pending()
                if step is None:
                    break

                if not self.policy.within_runtime(time.monotonic() - run_started):
                    job.status = WAITING
                    job.error = "Per-run runtime limit reached; resume to continue."
                    job.log.append(job.error)
                    self._persist_plan(job, plan)
                    return

                if getattr(step, "review_required", False):
                    job.status = WAITING
                    job.error = "This step needs explicit human review before it can run."
                    job.log.append(f"Waiting for explicit review: {step.title}.")
                    self._persist_plan(job, plan)
                    return

                allowed, reason = self.policy.can_run_autonomously(step.tool, step.args)
                if not allowed:
                    job.status = WAITING
                    job.error = reason
                    job.log.append(f"Waiting for approval/permission: {step.title} ({reason}).")
                    self._persist_plan(job, plan)
                    return

                index = plan.index_of(step)
                job.status = RUNNING
                job.error = ""
                job.log.append(f"Step {index + 1}/{len(plan.steps)}: {step.title}")
                self._persist_plan(job, plan)
                latest_job = self.store.load(job_id)
                if latest_job is not None and latest_job.cancel_requested:
                    latest_plan = engine.load(job.plan_id)
                    if latest_plan is not None and latest_plan.status == engine.COMPLETED:
                        job.cancel_requested = True
                        job.status = COMPLETED
                        job.result = latest_plan.summary()
                        job.error = ""
                        job.finished = _now()
                        job.log.append("Cancellation arrived after the final step completed.")
                        self._persist_plan(job, latest_plan)
                        return
                    engine.cancel(job.plan_id)
                    job.cancel_requested = True
                    job.status = CANCELLED
                    job.finished = _now()
                    job.log.append("Cancelled before the next action began.")
                    self._save_job_state(job)
                    return
                try:
                    task_agent._execute_step(
                        job.plan_id, index, player=None, speak=None,
                        should_stop=lambda jid=job_id: self._should_stop_job(jid))
                except Exception as exc:
                    latest_plan = engine.load(job.plan_id)
                    if latest_plan is not None and latest_plan.steps[index].status == engine.RUNNING:
                        engine.record_result(job.plan_id, index, ok=False,
                                             error=f"{type(exc).__name__}: {exc}")
                    else:
                        # A failure before the step was atomically claimed is
                        # not an action result; report it instead of looping.
                        raise

                plan = engine.load(job.plan_id)
                if plan is None:
                    raise RuntimeError("task plan disappeared during execution")
                self._persist_plan(job, plan)
                latest_job = self.store.load(job_id)
                if latest_job is not None and latest_job.cancel_requested:
                    if plan.status == engine.COMPLETED:
                        job.cancel_requested = True
                        job.status = COMPLETED
                        job.result = plan.summary()
                        job.error = ""
                        job.finished = _now()
                        job.log.append("Cancellation arrived after the final step completed.")
                        self._persist_plan(job, plan)
                        return
                    engine.cancel(job.plan_id)
                    job.cancel_requested = True
                    job.status = CANCELLED
                    job.finished = _now()
                    job.log.append("Cancelled after the current action returned.")
                    self._save_job_state(job)
                    return
                completed_step = plan.steps[index]
                if completed_step.status in (engine.DONE, engine.SKIPPED):
                    job.log.append(f"Verified result recorded for: {completed_step.title}")
                elif completed_step.status == engine.FAILED and plan.status == engine.ABANDONED:
                    replan_count = int(job.metadata.get("replan_attempts", 0) or 0)
                    job.log.append(
                        f"Step failed: {completed_step.title} — "
                        f"{(completed_step.error or 'no result')[:160]}")
                    if replan_count < self.policy.profile.max_replans:
                        # Spend and persist the bounded re-plan before calling the
                        # model so a crash cannot cause an unbounded planning loop.
                        job.metadata["replan_attempts"] = replan_count + 1
                        job.log.append(
                            f"Step failed after retries; requesting alternative plan "
                            f"({replan_count + 1}/{self.policy.profile.max_replans}).")
                        self._save_job_state(job)
                        registry = action_loader.get_registry()
                        tools = (registry.names() - {"task_agent", "jobs"}
                                 if registry is not None else set())
                        alternative = engine.replan_remaining(
                            job.plan_id, index, tools,
                            max_steps=self.policy.profile.max_plan_steps)
                        latest_job = self.store.load(job_id)
                        if latest_job is not None and latest_job.cancel_requested:
                            engine.cancel(job.plan_id)
                            job.cancel_requested = True
                            job.status = CANCELLED
                            job.finished = _now()
                            job.log.append("Cancelled after the re-plan request returned.")
                            self._save_job_state(job)
                            return
                        if alternative is not None:
                            job.error = ""
                            job.log.append("Alternative plan saved; continuing with eligible steps.")
                            self._persist_plan(job, alternative)
                            continue
                    job.status = FAILED
                    job.error = completed_step.error or "Step failed after retry/re-plan limit."
                    job.finished = _now()
                    job.log.append(f"Job failed: {job.error[:200]}")
                    self._save_job_state(job)
                    return

                if self._stop.is_set():
                    job.status = QUEUED
                    job.log.append("Paused at shutdown after saving the current step.")
                    self._persist_plan(job, plan)
                    return

            # Final state from the task engine is authoritative.
            plan = engine.load(job.plan_id) or plan
            self._persist_plan(job, plan)
            if plan.status == engine.COMPLETED:
                job.status = COMPLETED
                job.result = plan.summary()
                job.error = ""
                job.finished = _now()
                job.log.append("All planned steps completed.")
            elif plan.status == engine.CANCELLED:
                job.status = CANCELLED
                job.finished = _now()
            elif plan.status == engine.ABANDONED:
                job.status = FAILED
                job.error = job.error or plan.summary()
                job.finished = _now()
            else:
                job.status = WAITING
                job.error = job.error or "No eligible next step; review the plan."
            self._save_job_state(job)
        except Exception as exc:
            latest = self.store.load(job_id)
            if latest is not None:
                if latest.cancel_requested:
                    latest.status = CANCELLED
                    latest.error = "Cancelled after the current action returned."
                else:
                    latest.status = FAILED
                    latest.error = f"{type(exc).__name__}: {exc}"
                latest.finished = _now()
                latest.log.append(f"Worker stopped: {latest.error[:200]}")
                self._save_job_state(latest)
        finally:
            with _RUNNING_JOB_LOCK:
                _RUNNING_JOB_IDS.discard(job_id)
            with self._active_lock:
                self._active.discard(job_id)
                self._threads.pop(job_id, None)

    # -- status and control --
    def get_job(self, job_id: str) -> Optional[Job]:
        return self.store.load(job_id)

    def status(self, job_id: str) -> Optional[str]:
        job = self.store.load(job_id)
        return job.status if job else None

    def list_jobs(self, limit: int = 50) -> list[Job]:
        jobs = self.store.load_all()
        jobs.sort(key=lambda j: (j.created, j.id))
        count = max(0, int(limit))
        return jobs[-count:] if count else []

    def recent_history(self, limit: int = 20) -> list[str]:
        return [job.summary() for job in self.list_jobs(limit)]

    def logs(self, job_id: str) -> list[str]:
        job = self.store.load(job_id)
        return list(job.log) if job else []

    def report(self, job_id: str) -> str:
        job = self.store.load(job_id)
        if job is None:
            return "No such job."
        lines = [f"Job {job.id}: {job.status}", f"Goal: {job.goal}"]
        if job.plan_id:
            lines.append(f"Plan: {job.plan_id}")
            try:
                from core import task_engine as engine
                plan = engine.load(job.plan_id)
                if plan is None and job.plan_snapshot:
                    plan = engine._plan_from_dict(job.plan_snapshot)
                if plan is not None:
                    lines.append(engine.render(plan, policy=self.policy))
            except Exception as exc:
                lines.append(f"Plan report unavailable: {exc}")
        if job.result:
            lines.append(f"Result: {job.result}")
        if job.error:
            lines.append(f"Blocker/error: {job.error}")
        if job.log:
            lines.append("Job log:")
            lines.extend(f"  - {entry}" for entry in job.log[-20:])
        return "\n".join(lines)

    def cancel(self, job_id: str) -> bool:
        with _store_lock:
            job = self.store.load(job_id)
            if job is None or job.is_terminal():
                return False
            job.cancel_requested = True
            if job.status in (QUEUED, WAITING):
                job.status = CANCELLED
                job.finished = _now()
                job.log.append("Cancelled before the next action began.")
                try:
                    from core import task_engine as engine
                    if job.plan_id:
                        engine.cancel(job.plan_id)
                except Exception:
                    pass
            self.store.save(job)
            return True

    cancel_job = cancel

    def resume(self, job_id: str) -> bool:
        if not getattr(getattr(self.policy, "profile", None), "enable_background_jobs", True):
            return False
        if not self.is_running():
            self.start()
        with _store_lock:
            job = self.store.load(job_id)
            if job is None or job.is_terminal() or job.status == RUNNING:
                return False
            if job.status == WAITING:
                job.status = QUEUED
                job.cancel_requested = False
                job.error = ""
                job.log.append("Resumed by request.")
                self.store.save(job)
        if self.synchronous:
            self._run_job(job_id)
        else:
            self._dispatch(job_id)
        return True

    resume_job = resume

    def wait_for(self, job_id: str, timeout: float = 30.0) -> Optional[Job]:
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            job = self.store.load(job_id)
            if job is None or job.is_terminal() or job.status == WAITING:
                return job
            time.sleep(0.05)
        return self.store.load(job_id)

    def drain(self, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            if not any(job.status in (QUEUED, RUNNING)
                       for job in self.store.load_all()):
                return
            time.sleep(0.05)

    # Backwards-friendly names for callers which use task/job terminology.
    get = get_job
    get_status = status
    list = list_jobs
    history = recent_history


# Application-wide scheduler singleton. All built-in entry points use it.
_GLOBAL_WORKER: Optional[BackgroundWorker] = None
_GLOBAL_LOCK = threading.Lock()


def get_worker() -> BackgroundWorker:
    global _GLOBAL_WORKER
    with _GLOBAL_LOCK:
        if _GLOBAL_WORKER is None:
            _GLOBAL_WORKER = BackgroundWorker()
        return _GLOBAL_WORKER


def set_worker(worker: Optional[BackgroundWorker]) -> None:
    global _GLOBAL_WORKER
    with _GLOBAL_LOCK:
        _GLOBAL_WORKER = worker


def start_worker(store_path=None, **kwargs) -> BackgroundWorker:
    global _GLOBAL_WORKER
    if store_path is None:
        worker = get_worker()
    else:
        worker = BackgroundWorker(store_path=store_path, **kwargs)
        with _GLOBAL_LOCK:
            previous = _GLOBAL_WORKER
            _GLOBAL_WORKER = worker
        if previous is not None and previous is not worker:
            previous.stop()
    worker.start()
    return worker


def stop_worker(timeout: float = 10.0) -> None:
    global _GLOBAL_WORKER
    with _GLOBAL_LOCK:
        worker = _GLOBAL_WORKER
    if worker is not None:
        worker.stop(timeout=timeout)


def submit_job(goal: str, **kwargs) -> Job:
    return get_worker().submit(goal, **kwargs)


def submit(goal: str, **kwargs) -> Job:
    return submit_job(goal, **kwargs)


def enqueue(goal: str, **kwargs) -> Job:
    return submit_job(goal, **kwargs)


def get_job(job_id: str) -> Optional[Job]:
    try:
        job = get_worker().get_job(job_id)
        if job is not None:
            return job
    except Exception:
        pass
    return load(job_id)


def cancel_job(job_id: str) -> bool:
    return get_worker().cancel(job_id)


def cancel(job_id: str) -> bool:
    return cancel_job(job_id)


def resume_job(job_id: str) -> bool:
    return get_worker().resume(job_id)


def resume(job_id: str) -> bool:
    return resume_job(job_id)


def list_jobs(limit: int = 50) -> list[Job]:
    return get_worker().list_jobs(limit)


def recover_jobs() -> list[str]:
    return get_worker().recover()


__all__ = [
    "QUEUED", "RUNNING", "WAITING", "COMPLETED", "FAILED", "CANCELLED",
    "ALL_STATES", "TERMINAL_STATES", "Job", "JobStore", "BackgroundWorker",
    "set_store_path", "get_store_path", "load_all", "load", "save", "delete",
    "get_worker", "set_worker", "start_worker", "stop_worker",
    "submit_job", "submit", "enqueue", "get_job", "cancel_job", "cancel",
    "resume_job", "resume", "list_jobs", "recover_jobs",
]
