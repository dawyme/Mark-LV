"""actions/jobs.py — voice/UI interface to the background job worker."""
from __future__ import annotations

from core import background_worker as bw


def _worker():
    w = bw.get_worker()
    try:
        if not w.is_running():
            w.start()
    except Exception:
        pass
    return w


def jobs(parameters: dict, player=None, speak=None, session_memory=None) -> str:
    params = parameters or {}
    action = str(params.get("action", "list")).strip().lower()
    goal = str(params.get("goal", "") or "").strip()
    job_id = str(params.get("job_id", "") or "").strip()

    w = _worker()

    try:
        if action in ("submit", "start", "new", "create", "run"):
            if not goal:
                return "Tell me what you want done in the background."
            job = w.submit(goal)
            return f"Queued background job {job.id}: {job.goal}"

        if action in ("status", "progress", "show"):
            if job_id:
                return w.report(job_id)
            jobs = w.list_jobs(10)
            if not jobs:
                return "No background jobs."
            return "\n".join(j.summary() for j in jobs)

        if action in ("list", "history"):
            jobs = w.list_jobs(10)
            if not jobs:
                return "No background jobs."
            return "\n".join(j.summary() for j in jobs)

        if action in ("cancel", "stop", "abandon"):
            if not job_id:
                # cancel most recent active job
                active = [j for j in w.list_jobs(50) if j.status in ("queued", "running", "waiting")]
                if not active:
                    return "No active job to cancel."
                job_id = active[-1].id
            before = w.get_job(job_id)
            if before is None:
                return "No such job."
            if not w.cancel(job_id):
                latest = w.get_job(job_id)
                return (f"Job {job_id} is already {latest.status}." if latest
                        else "No such job.")
            latest = w.get_job(job_id)
            if latest is not None and latest.status == bw.CANCELLED:
                return f"Cancelled job {job_id}."
            if latest is not None and latest.cancel_requested:
                return (f"Cancellation requested for job {job_id}; its active action "
                        "will finish before more work is stopped.")
            return f"Job {job_id} is now {latest.status if latest else 'unavailable'}."

        if action in ("resume", "continue"):
            if not job_id:
                waiting = [j for j in w.list_jobs(50) if j.status == "waiting"]
                if not waiting:
                    return "No waiting job to resume."
                job_id = waiting[-1].id
            ok = w.resume(job_id)
            return f"Resumed job {job_id}." if ok else "Could not resume that job."

        if action in ("logs", "log"):
            if not job_id:
                return "Which job? Give me a job_id."
            logs = w.logs(job_id)
            if not logs:
                return "No log entries yet."
            return "\n".join(logs[-20:])

        if action in ("report",):
            if not job_id:
                return "Which job? Give me a job_id."
            return w.report(job_id)

        if action in ("recover",):
            recovered = w.recover()
            return f"Recovered {len(recovered)} job(s)."

        return ("Say which of these you want: submit, status, list, cancel, "
                "resume, logs, report, or recover.")
    except Exception as e:
        return f"Job action failed: {e}"


TOOL = {
    "name": "jobs",
    "description": (
        "Manage persistent background jobs: submit a goal to run in the "
        "background, check status, list recent jobs, cancel, resume, or read "
        "logs. Jobs persist across restarts and run off the voice/UI threads."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {
                "type": "STRING",
                "description": "submit | status | list | cancel | resume | logs | report | recover",
            },
            "goal": {
                "type": "STRING",
                "description": "Goal to run in the background (for action=submit)",
            },
            "job_id": {
                "type": "STRING",
                "description": "A specific job id; omit for the most recent relevant one",
            },
        },
        "required": ["action"],
    },
    "handler": jobs,
}
