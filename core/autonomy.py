"""Centralized, testable autonomy policy for Mark-LV.

A small allow-list keeps routine low-impact work autonomous while new or
side-effecting tools default to a human checkpoint. Profiles can be stored in
``config/autonomy.json``; the safe defaults work without creating that file.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from core import task_engine as _engine
    _DEFAULT_SAFE_TOOLS = set(getattr(_engine, "DEFAULT_ALLOWED_TOOLS", frozenset())) - {
        "weather_report",  # launches a browser; use web_search for headless weather
        "flight_finder",  # opens Google Flights in the browser and can save to Desktop
    }
except Exception:  # pragma: no cover - defensive import fallback
    _DEFAULT_SAFE_TOOLS = {"think"}

SAFE_RISK = "safe"
SENSITIVE_RISK = "sensitive"
DESTRUCTIVE_RISK = "destructive"
EXTERNAL_RISK = "external"
SECURITY_RISK = "security"

# These are safeguards, not an exhaustive definition of capabilities. Anything
# not explicitly trusted by the profile remains confirmation-gated.
_DESTRUCTIVE_TOOLS = frozenset({
    "file_controller", "computer_settings", "shutdown_jarvis", "desktop",
    "dev_agent", "game_updater",
})
_EXTERNAL_TOOLS = frozenset({
    "send_message", "email", "gmail_control", "calendar_control",
    "whatsapp_send", "slack_send", "post_to_social", "browser_control",
})
_SECURITY_TOOLS = frozenset({"credential_manager", "password_manager", "security_settings"})
_SENSITIVE_TOOLS = frozenset({
    "weather_report", "flight_finder", "youtube_video", "computer_control", "reminder",
})
_NETWORK_TOOLS = frozenset({
    "web_search", "research", "weather_report", "flight_finder",
    "browser_control", "youtube_video", "send_message", "gmail_control",
    "calendar_control", "file_processor", "code_helper",
})
_WRITE_TOOLS = frozenset({"dev_agent"})
_FILE_PROCESSOR_READ_ACTIONS = frozenset({
    "analyze", "describe", "explain", "info", "list", "ocr", "read",
    "review", "stats", "summarize", "transcribe", "validate", "word_count",
})
_FILE_CONTROLLER_READ_ACTIONS = frozenset({
    "disk_usage", "find", "info", "largest", "list", "read",
})
_FILE_CONTROLLER_WRITE_ACTIONS = frozenset({
    "copy", "create_file", "create_folder", "write",
})
_FILE_CONTROLLER_KNOWN_ACTIONS = (
    _FILE_CONTROLLER_READ_ACTIONS | _FILE_CONTROLLER_WRITE_ACTIONS
    | frozenset({"delete", "move", "organize_desktop", "rename"})
)
_SAFE_COMPUTER_SETTINGS_ACTIONS = frozenset({
    "brightness_down", "brightness_up", "dark_mode", "mute", "toggle_mute",
    "unmute", "volume_down", "volume_set", "volume_up",
})
_ALWAYS_CONFIRM_COMPUTER_ACTIONS = frozenset({
    "close_app", "close_window", "lock_screen", "restart", "shutdown", "toggle_wifi",
})


def _base_dir() -> Path:
    import sys
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


@dataclass
class AutonomyProfile:
    """Operational limits and permissions for background task execution.

    ``max_retries`` counts retries after the initial attempt (so 1 means at
    most two attempts for a step). Runtime is a per-worker-run ceiling; a
    paused/waiting job can be resumed in a later run.
    """

    name: str = "autonomous-routine"
    allow_autonomous: bool = True
    enable_background_jobs: bool = True
    autonomous_tools: set[str] = field(default_factory=lambda: set(_DEFAULT_SAFE_TOOLS))
    max_job_runtime_seconds: int = 300
    max_concurrent_jobs: int = 2
    max_queued_jobs: int = 100
    max_plan_steps: int = 6
    max_retries: int = 1
    max_replans: int = 1
    retry_backoff_seconds: float = 1.0
    require_confirmation_for: set[str] = field(default_factory=lambda: {
        SENSITIVE_RISK, DESTRUCTIVE_RISK, EXTERNAL_RISK, SECURITY_RISK
    })
    allow_network_actions: bool = True
    allow_file_writes: bool = False
    allow_destructive: bool = False
    allow_external_facing: bool = False
    respect_os_permissions: bool = True

    def __post_init__(self):
        self.name = str(self.name or "autonomous-routine")[:80]
        self.allow_autonomous = _bounded_bool(self.allow_autonomous, True)
        self.enable_background_jobs = _bounded_bool(self.enable_background_jobs, True)
        self.allow_network_actions = _bounded_bool(self.allow_network_actions, True)
        self.allow_file_writes = _bounded_bool(self.allow_file_writes, False)
        self.allow_destructive = _bounded_bool(self.allow_destructive, False)
        self.allow_external_facing = _bounded_bool(self.allow_external_facing, False)
        self.respect_os_permissions = _bounded_bool(self.respect_os_permissions, True)
        self.autonomous_tools = {str(t).strip().lower() for t in (self.autonomous_tools or set())
                                 if str(t).strip()} | {"think"}
        self.require_confirmation_for = {
            str(r).strip().lower() for r in (self.require_confirmation_for or set())
            if str(r).strip()
        }
        self.max_job_runtime_seconds = _bounded_int(self.max_job_runtime_seconds, 1, 86400, 300)
        self.max_concurrent_jobs = _bounded_int(self.max_concurrent_jobs, 1, 16, 2)
        self.max_queued_jobs = _bounded_int(self.max_queued_jobs, 1, 10000, 100)
        self.max_plan_steps = _bounded_int(self.max_plan_steps, 1, 6, 6)
        self.max_retries = _bounded_int(self.max_retries, 0, 10, 1)
        self.max_replans = _bounded_int(self.max_replans, 0, 3, 1)
        try:
            self.retry_backoff_seconds = min(60.0, max(0.0, float(self.retry_backoff_seconds)))
        except (TypeError, ValueError):
            self.retry_backoff_seconds = 1.0


def _bounded_int(value, low: int, high: int, default: int) -> int:
    try:
        return min(high, max(low, int(value)))
    except (TypeError, ValueError):
        return default


def _bounded_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
    return default


class AutonomyPolicy:
    def __init__(self, profile: Optional[AutonomyProfile] = None):
        self.profile = profile or AutonomyProfile()

    @property
    def max_job_runtime_seconds(self) -> int:
        return self.profile.max_job_runtime_seconds

    @property
    def max_concurrent_jobs(self) -> int:
        return self.profile.max_concurrent_jobs

    @property
    def max_retries(self) -> int:
        return self.profile.max_retries

    @property
    def retry_backoff_seconds(self) -> float:
        return self.profile.retry_backoff_seconds

    @property
    def max_replans(self) -> int:
        return self.profile.max_replans

    def classify_tool(self, tool: str) -> str:
        tool = (tool or "").strip().lower()
        if not tool:
            return SENSITIVE_RISK
        if tool in _SECURITY_TOOLS:
            return SECURITY_RISK
        if tool in _SENSITIVE_TOOLS:
            return SENSITIVE_RISK
        if tool in _DESTRUCTIVE_TOOLS or tool in _WRITE_TOOLS:
            return DESTRUCTIVE_RISK
        if tool in _EXTERNAL_TOOLS:
            return EXTERNAL_RISK
        if tool in self.profile.autonomous_tools:
            return SAFE_RISK
        return SENSITIVE_RISK

    risk_of = classify_tool

    @staticmethod
    def _file_processor_is_read_only(args: dict | None) -> bool:
        args = args if isinstance(args, dict) else {}
        action = str(args.get("action", "") or "").strip().lower()
        # file_processor is a broad multi-operation tool. Automatic execution
        # requires a named read-only action and disables its optional output
        # save behavior; omitted/"auto" actions fail closed.
        return action in _FILE_PROCESSOR_READ_ACTIONS and args.get("save") is False

    @staticmethod
    def _code_helper_is_read_only(args: dict | None) -> bool:
        args = args if isinstance(args, dict) else {}
        return str(args.get("action", "") or "").strip().lower() == "explain"

    @staticmethod
    def _file_controller_target(args: dict, *, require_name: bool = True) -> Optional[Path]:
        name = str(args.get("name", "") or "").strip()
        if require_name and not name:
            return None
        try:
            from actions.file_controller import _resolve_path
            base = _resolve_path(str(args.get("path", "desktop") or "desktop"))
            return (base / name) if name else base
        except Exception:
            return None

    @staticmethod
    def _file_controller_action_known(args: dict | None) -> bool:
        args = args if isinstance(args, dict) else {}
        action = str(args.get("action", "") or "").strip().lower()
        return action in _FILE_CONTROLLER_KNOWN_ACTIONS

    @classmethod
    def _file_controller_is_routine(cls, args: dict | None) -> bool:
        args = args if isinstance(args, dict) else {}
        action = str(args.get("action", "") or "").strip().lower()
        if action in _FILE_CONTROLLER_READ_ACTIONS:
            return True
        if action == "write":
            # Append is non-destructive; replacing existing contents is gated.
            return args.get("append") is True
        if action == "create_file":
            target = cls._file_controller_target(args)
            return target is not None and not target.exists()
        if action == "create_folder":
            target = cls._file_controller_target(args, require_name=False)
            return target is not None and (not target.exists() or target.is_dir())
        return False

    @staticmethod
    def _open_app_is_routine(args: dict | None) -> bool:
        args = args if isinstance(args, dict) else {}
        name = str(args.get("app_name", "") or "").strip().lower()
        if not name:
            return False
        try:
            from actions.open_app import _APP_ALIASES
            return name in _APP_ALIASES
        except Exception:
            return False

    @staticmethod
    def _computer_settings_action_known(args: dict | None) -> bool:
        args = args if isinstance(args, dict) else {}
        action = str(args.get("action", "") or "").strip().lower()
        if not action:
            return False
        try:
            from actions.computer_settings import ACTION_MAP, _VALUE_ACTIONS
            return action in set(ACTION_MAP) | set(_VALUE_ACTIONS)
        except Exception:
            return False

    @staticmethod
    def _computer_settings_is_routine(args: dict | None) -> bool:
        args = args if isinstance(args, dict) else {}
        action = str(args.get("action", "") or "").strip().lower()
        return action in _SAFE_COMPUTER_SETTINGS_ACTIONS

    @staticmethod
    def _file_controller_writes(args: dict | None) -> bool:
        args = args if isinstance(args, dict) else {}
        return str(args.get("action", "") or "").strip().lower() in _FILE_CONTROLLER_WRITE_ACTIONS

    def requires_confirmation(self, tool: str, args: dict | None = None) -> tuple[bool, str]:
        """Whether a human checkpoint is required, independent of job mode."""
        tool = (tool or "").strip().lower()
        risk = self.classify_tool(tool)
        args_dict = args if isinstance(args, dict) else {}
        file_write = (tool in _WRITE_TOOLS
                      or (tool == "file_controller" and self._file_controller_writes(args))
                      or (tool == "flight_finder" and bool(args_dict.get("save", False))))

        if tool == "file_processor" and not self._file_processor_is_read_only(args):
            return True, "file processing action may write or execute; explicit confirmation required"
        if tool == "code_helper" and not self._code_helper_is_read_only(args):
            return True, "code action may write or execute; explicit confirmation required"

        # Preserve Mark-LV's operation-aware defaults for ordinary local
        # actions while failing closed on missing or unknown operations.
        if tool == "file_controller":
            if not self._file_controller_action_known(args):
                return True, "unknown file operation requires explicit confirmation"
            if self._file_controller_is_routine(args):
                risk = SAFE_RISK if tool in self.profile.autonomous_tools else SENSITIVE_RISK
        elif tool == "computer_settings":
            if not self._computer_settings_action_known(args):
                return True, "unknown computer setting requires explicit confirmation"
            action = str((args or {}).get("action", "") or "").strip().lower()
            if action in _ALWAYS_CONFIRM_COMPUTER_ACTIONS:
                return True, "high-impact computer control always requires explicit confirmation"
            if self._computer_settings_is_routine(args):
                risk = SAFE_RISK if tool in self.profile.autonomous_tools else SENSITIVE_RISK
        elif tool == "open_app":
            if not self._open_app_is_routine(args):
                return True, "unknown application launch requires explicit confirmation"
            risk = SAFE_RISK if tool in self.profile.autonomous_tools else SENSITIVE_RISK

        if risk == SECURITY_RISK:
            return True, "security-sensitive operation requires confirmation"
        if risk == DESTRUCTIVE_RISK:
            if not self.profile.allow_destructive:
                return True, "destructive or write operation requires confirmation"
            if file_write and not self.profile.allow_file_writes:
                return True, "file writes are disabled by profile"
        if risk == EXTERNAL_RISK and not self.profile.allow_external_facing:
            return True, "external-facing operation requires confirmation"
        if risk in self.profile.require_confirmation_for:
            return True, f"{risk} operation requires confirmation by profile"
        if tool in _NETWORK_TOOLS and not self.profile.allow_network_actions:
            return True, "network actions are disabled by profile"
        if tool not in self.profile.autonomous_tools:
            return True, f"tool '{tool}' is not in the autonomous allow-list"
        if file_write and not self.profile.allow_file_writes:
            return True, "file writes are disabled by profile"
        if (tool in _WRITE_TOOLS and self.profile.respect_os_permissions
                and not self.detect_os_permissions().get("writable_cwd", False)):
            return True, "current working directory is not writable"
        return False, "no confirmation required"

    def can_run_autonomously(self, tool: str, args: dict | None = None) -> tuple[bool, str]:
        """Return ``(allowed, reason)``. Unknown tools fail closed."""
        if not self.profile.allow_autonomous or not self.profile.enable_background_jobs:
            return False, "autonomous background execution is disabled by profile"
        needs_confirmation, reason = self.requires_confirmation(tool, args)
        if needs_confirmation:
            return False, reason
        return True, "routine operation allowed"

    def can_replay_after_crash(self, tool: str, args: dict | None = None) -> bool:
        """Only known read-only built-ins may be retried after an unknown outcome.

        Even a tool explicitly enabled for autonomous use is not assumed
        idempotent when it can write or act externally.
        """
        tool = (tool or "").strip().lower()
        if (tool not in _DEFAULT_SAFE_TOOLS or tool in _WRITE_TOOLS
                or tool in _EXTERNAL_TOOLS
                or tool in {"open_app", "computer_settings", "research"}):
            return False
        if tool in _SENSITIVE_TOOLS:
            return False
        if tool == "file_controller":
            args = args if isinstance(args, dict) else {}
            action = str(args.get("action", "") or "").strip().lower()
            if action not in _FILE_CONTROLLER_READ_ACTIONS:
                return False
        allowed, _ = self.can_run_autonomously(tool, args)
        return allowed

    def check_submit(self, current_job_count: int) -> tuple[bool, str]:
        if not self.profile.enable_background_jobs:
            return False, "background jobs are disabled by profile"
        if current_job_count >= self.profile.max_queued_jobs:
            return False, "queued-job limit reached"
        return True, "ok"

    def check_concurrency(self, active_count: int) -> tuple[bool, str]:
        if active_count >= self.profile.max_concurrent_jobs:
            return False, "concurrency limit reached"
        return True, "ok"

    def within_runtime(self, elapsed_seconds: float) -> bool:
        return elapsed_seconds <= self.profile.max_job_runtime_seconds

    @classmethod
    def from_dict(cls, data: dict) -> "AutonomyPolicy":
        if not isinstance(data, dict):
            data = {}
        values = data.get("profile", data)
        if not isinstance(values, dict):
            values = {}
        allowed = {
            "name", "allow_autonomous", "enable_background_jobs",
            "max_job_runtime_seconds", "max_concurrent_jobs", "max_queued_jobs",
            "max_plan_steps", "max_retries", "max_replans", "retry_backoff_seconds",
            "allow_network_actions", "allow_file_writes", "allow_destructive",
            "allow_external_facing", "respect_os_permissions",
        }
        kwargs = {key: values[key] for key in allowed if key in values}
        for key in ("autonomous_tools", "require_confirmation_for"):
            val = values.get(key)
            if isinstance(val, (list, tuple, set)):
                kwargs[key] = set(val)
        try:
            return cls(AutonomyProfile(**kwargs))
        except Exception:
            return cls()

    @classmethod
    def load(cls, path=None) -> "AutonomyPolicy":
        path = Path(path) if path is not None else _base_dir() / "config" / "autonomy.json"
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls.from_dict(data)
        except Exception as e:
            print(f"[Autonomy] Could not load {path.name}; using safe defaults: {e}")
            return cls()

    def save(self, path=None) -> None:
        path = Path(path) if path is not None else _base_dir() / "config" / "autonomy.json"
        p = self.profile
        data = {"profile": {
            "name": p.name,
            "allow_autonomous": p.allow_autonomous,
            "enable_background_jobs": p.enable_background_jobs,
            "autonomous_tools": sorted(p.autonomous_tools),
            "max_job_runtime_seconds": p.max_job_runtime_seconds,
            "max_concurrent_jobs": p.max_concurrent_jobs,
            "max_queued_jobs": p.max_queued_jobs,
            "max_plan_steps": p.max_plan_steps,
            "max_retries": p.max_retries,
            "max_replans": p.max_replans,
            "retry_backoff_seconds": p.retry_backoff_seconds,
            "require_confirmation_for": sorted(p.require_confirmation_for),
            "allow_network_actions": p.allow_network_actions,
            "allow_file_writes": p.allow_file_writes,
            "allow_destructive": p.allow_destructive,
            "allow_external_facing": p.allow_external_facing,
            "respect_os_permissions": p.respect_os_permissions,
        }}
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

    @staticmethod
    def detect_os_permissions() -> dict:
        try:
            uid = os.getuid()
            is_admin = uid == 0
        except AttributeError:
            uid, is_admin = None, False
        try:
            writable = os.access(Path.cwd(), os.W_OK)
        except Exception:
            writable = False
        return {"uid": uid, "is_admin": is_admin, "writable_cwd": writable}

    def describe(self) -> str:
        p = self.profile
        return (f"Autonomy profile '{p.name}': enabled={p.allow_autonomous and p.enable_background_jobs}, "
                f"tools={sorted(p.autonomous_tools)}, max_runtime={p.max_job_runtime_seconds}s, "
                f"max_concurrency={p.max_concurrent_jobs}, queued_limit={p.max_queued_jobs}, "
                f"retries={p.max_retries}")


__all__ = [
    "AutonomyProfile", "AutonomyPolicy", "SAFE_RISK", "SENSITIVE_RISK",
    "DESTRUCTIVE_RISK", "EXTERNAL_RISK", "SECURITY_RISK",
]
