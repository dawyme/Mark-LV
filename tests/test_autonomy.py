"""Central autonomy profile enforcement and local config behavior."""
import json

from core.autonomy import (
    AutonomyPolicy, AutonomyProfile, DESTRUCTIVE_RISK, EXTERNAL_RISK,
    SAFE_RISK, SECURITY_RISK, SENSITIVE_RISK,
)


def test_default_profile_allows_only_known_routine_tools():
    policy = AutonomyPolicy()

    assert policy.classify_tool("web_search") == SAFE_RISK
    assert policy.can_run_autonomously("web_search")[0]
    assert policy.can_run_autonomously("think")[0]
    assert not policy.can_run_autonomously("new_unregistered_tool")[0]
    assert policy.classify_tool("new_unregistered_tool") == SENSITIVE_RISK


def test_broad_file_and_code_tools_require_explicit_read_only_arguments():
    policy = AutonomyPolicy()

    assert not policy.can_run_autonomously("file_processor", {
        "action": "summarize", "file_path": "doc.pdf"
    })[0]
    assert policy.can_run_autonomously("file_processor", {
        "action": "summarize", "file_path": "doc.pdf", "save": False
    })[0]
    assert not policy.can_run_autonomously("file_processor", {
        "action": "extract_text", "file_path": "doc.pdf", "save": False
    })[0]
    assert policy.can_run_autonomously("code_helper", {
        "action": "explain", "file_path": "script.py"
    })[0]
    assert not policy.can_run_autonomously("code_helper", {
        "action": "run", "file_path": "script.py"
    })[0]


def test_browser_and_desktop_actions_require_confirmation():
    policy = AutonomyPolicy()

    assert "flight_finder" not in policy.profile.autonomous_tools
    for tool in ("weather_report", "flight_finder", "youtube_video",
                 "computer_control", "reminder"):
        assert not policy.can_run_autonomously(tool)[0]
        assert policy.requires_confirmation(tool)[0]


def test_operation_aware_file_computer_and_known_app_defaults(tmp_path, monkeypatch):
    from actions import file_controller

    monkeypatch.setattr(file_controller, "_resolve_path", lambda _raw: tmp_path)
    policy = AutonomyPolicy()

    assert policy.can_run_autonomously("file_controller", {
        "action": "list", "path": str(tmp_path),
    })[0]
    assert policy.can_run_autonomously("file_controller", {
        "action": "read", "path": str(tmp_path), "name": "notes.txt",
    })[0]
    assert policy.can_run_autonomously("computer_settings", {"action": "volume_down"})[0]
    assert policy.can_run_autonomously("open_app", {"app_name": "calculator"})[0]

    # New file creation remains opt-in via the centralized write permission;
    # overwriting an existing file stays destructive even when writes are on.
    create = {"action": "create_file", "path": str(tmp_path), "name": "report.txt"}
    assert not policy.can_run_autonomously("file_controller", create)[0]
    writes_enabled = AutonomyPolicy(AutonomyProfile(allow_file_writes=True))
    assert writes_enabled.can_run_autonomously("file_controller", create)[0]
    (tmp_path / "report.txt").write_text("existing", encoding="utf-8")
    assert not writes_enabled.can_run_autonomously("file_controller", create)[0]

    sensitive_computer_actions = (
        "close_app", "close_window", "lock_screen", "restart", "shutdown", "toggle_wifi",
    )
    for action in sensitive_computer_actions:
        assert not policy.can_run_autonomously("computer_settings", {"action": action})[0]
    explicit_destructive_profile = AutonomyPolicy(AutonomyProfile(
        allow_destructive=True, allow_file_writes=True, require_confirmation_for=set()))
    for action in sensitive_computer_actions:
        assert not explicit_destructive_profile.can_run_autonomously(
            "computer_settings", {"action": action})[0]
    assert not explicit_destructive_profile.can_run_autonomously(
        "file_controller", {"action": "invented"})[0]
    assert not policy.can_run_autonomously("open_app", {"app_name": "unlisted-command"})[0]
    assert not policy.can_run_autonomously("file_controller", {"action": "invented"})[0]
    assert not policy.can_replay_after_crash("file_controller", create)
    assert not policy.can_replay_after_crash("computer_settings", {"action": "volume_down"})
    assert not policy.can_replay_after_crash("open_app", {"app_name": "calculator"})


def test_flight_report_file_write_uses_the_central_write_permission():
    profile = AutonomyProfile(
        autonomous_tools={"think", "flight_finder"},
        require_confirmation_for=set(),
        allow_network_actions=True,
        allow_file_writes=False,
    )
    policy = AutonomyPolicy(profile)

    assert policy.can_run_autonomously("flight_finder", {"save": False})[0]
    assert not policy.can_run_autonomously("flight_finder", {"save": True})[0]
    profile.allow_file_writes = True
    assert policy.can_run_autonomously("flight_finder", {"save": True})[0]


def test_destructive_external_and_security_operations_require_human_approval():
    policy = AutonomyPolicy()

    assert policy.classify_tool("file_controller") == DESTRUCTIVE_RISK
    assert policy.requires_confirmation("file_controller")[0]
    assert policy.classify_tool("send_message") == EXTERNAL_RISK
    assert policy.requires_confirmation("send_message")[0]
    assert policy.classify_tool("password_manager") == SECURITY_RISK
    assert policy.requires_confirmation("password_manager")[0]


def test_network_actions_can_be_disabled_in_one_central_policy():
    profile = AutonomyProfile(allow_network_actions=False)
    policy = AutonomyPolicy(profile)

    assert not policy.can_run_autonomously("web_search")[0]
    assert policy.requires_confirmation("web_search")[0]
    # The broad file/code tools may send content to the configured model
    # provider, so network-off configuration requires approval even for a
    # nominally read-only operation.
    assert not policy.can_run_autonomously("file_processor", {
        "action": "info", "file_path": "report.pdf", "save": False
    })[0]
    assert not policy.can_run_autonomously("code_helper", {
        "action": "explain", "code": "print('hello')"
    })[0]
    assert not policy.can_run_autonomously("file_processor", {
        "action": "convert", "file_path": "report.pdf", "save": False
    })[0]


def test_unknown_tools_fail_closed_when_not_explicitly_allowed():
    profile = AutonomyProfile(autonomous_tools={"think", "web_search"})
    policy = AutonomyPolicy(profile)

    allowed, reason = policy.can_run_autonomously("unknown")
    assert not allowed
    assert "confirmation" in reason or "autonomous allow-list" in reason


def test_external_actions_need_both_an_explicit_permission_and_allow_list_entry():
    profile = AutonomyProfile(
        autonomous_tools={"think", "send_message"},
        allow_external_facing=True,
        allow_network_actions=True,
        require_confirmation_for={"sensitive", "destructive", "security"},
    )
    policy = AutonomyPolicy(profile)

    assert policy.can_run_autonomously("send_message")[0]
    assert not AutonomyPolicy(AutonomyProfile(
        autonomous_tools={"think", "send_message"},
        allow_external_facing=False,
    )).can_run_autonomously("send_message")[0]


def test_confirmation_risk_classes_are_configurable_but_permissions_still_apply():
    profile = AutonomyProfile(
        autonomous_tools={"think", "file_controller"},
        allow_destructive=True,
        allow_file_writes=True,
        require_confirmation_for={"external", "security"},
    )
    policy = AutonomyPolicy(profile)

    assert policy.can_run_autonomously("file_controller", {"action": "list"})[0]
    assert not AutonomyPolicy(AutonomyProfile(
        autonomous_tools={"think", "file_controller"},
        allow_destructive=False,
        allow_file_writes=True,
        require_confirmation_for={"external", "security"},
    )).can_run_autonomously("file_controller", {
        "action": "delete", "path": "desktop", "name": "report.txt"
    })[0]


def test_crash_replay_is_limited_to_known_read_only_tools():
    profile = AutonomyProfile(
        autonomous_tools={"think", "web_search", "send_message"},
        allow_external_facing=True,
    )
    policy = AutonomyPolicy(profile)

    assert policy.can_replay_after_crash("web_search")
    assert not policy.can_replay_after_crash("research")  # stores a brief in local memory
    assert not policy.can_replay_after_crash("send_message")
    assert not policy.can_replay_after_crash("file_controller")


def test_profile_limits_are_bounded_and_saved_atomically(tmp_path):
    path = tmp_path / "autonomy.json"
    profile = AutonomyProfile(max_concurrent_jobs=999, max_plan_steps=100,
                              max_retries=50, max_replans=20,
                              retry_backoff_seconds=1000)
    policy = AutonomyPolicy(profile)

    assert profile.max_concurrent_jobs == 16
    assert profile.max_plan_steps == 6
    assert profile.max_retries == 10
    assert profile.max_replans == 3
    assert profile.retry_backoff_seconds == 60
    policy.save(path)
    loaded = AutonomyPolicy.load(path)

    assert loaded.profile.max_concurrent_jobs == 16
    assert loaded.profile.max_plan_steps == 6
    assert loaded.profile.max_retries == 10
    assert loaded.profile.max_replans == 3
    assert json.loads(path.read_text(encoding="utf-8"))["profile"]["allow_autonomous"] is True


def test_string_booleans_in_local_config_are_parsed_fail_safe():
    policy = AutonomyPolicy.from_dict({"profile": {
        "allow_autonomous": "false",
        "enable_background_jobs": "false",
        "allow_destructive": "false",
        "allow_network_actions": "off",
        "allow_file_writes": "true",
    }})

    assert policy.profile.allow_autonomous is False
    assert policy.profile.enable_background_jobs is False
    assert policy.profile.allow_destructive is False
    assert policy.profile.allow_network_actions is False
    assert policy.profile.allow_file_writes is True
    assert not policy.can_run_autonomously("web_search")[0]


def test_corrupt_profile_falls_back_to_defaults_without_raising(tmp_path):
    path = tmp_path / "autonomy.json"
    path.write_text("{bad", encoding="utf-8")

    policy = AutonomyPolicy.load(path)

    assert policy.profile.enable_background_jobs is True
    assert policy.profile.autonomous_tools


def test_os_permission_detection_does_not_claim_admin_privileges():
    info = AutonomyPolicy.detect_os_permissions()

    assert "is_admin" in info and isinstance(info["is_admin"], bool)
    assert "writable_cwd" in info and isinstance(info["writable_cwd"], bool)
