"""Background job additions keep the existing action/plugin contract intact."""
import json
import sys
from pathlib import Path

from core import action_loader, plugin_loader
from memory import config_manager


def test_action_registry_still_discovers_and_dispatches_the_tool_contract(tmp_path):
    actions_dir = tmp_path / "actions"
    actions_dir.mkdir()
    (actions_dir / "sample_action.py").write_text(
        "def run(parameters):\n    return 'action result'\n"
        "TOOL = {'name': 'sample_action', 'description': 'sample', "
        "'parameters': {'type': 'OBJECT', 'properties': {}}, 'handler': run}\n",
        encoding="utf-8",
    )
    logs = []

    registry = action_loader.discover_actions(actions_dir, logger=logs.append)

    assert registry.has("sample_action")
    assert registry.run("sample_action", {}) == "action result"
    assert "sample_action" in registry.names()
    # An empty result stays empty for task result verification; main.py retains
    # its historical "Done." fallback at the conversational dispatch edge.
    (actions_dir / "empty_action.py").write_text(
        "def run(parameters):\n    return ''\n"
        "TOOL = {'name': 'empty_action', 'description': 'empty', "
        "'parameters': {'type': 'OBJECT', 'properties': {}}, 'handler': run}\n",
        encoding="utf-8",
    )
    registry = action_loader.discover_actions(actions_dir, logger=lambda _line: None)
    assert registry.run("empty_action", {}) == ""


def test_plugin_discovery_and_existing_plugin_config_remain_compatible(tmp_path, monkeypatch):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "sample_plugin.py").write_text(
        "PLUGIN = {'name': 'sample_plugin', 'description': 'sample', "
        "'parameters': {'type': 'OBJECT', 'properties': {}}}\n"
        "def run(parameters):\n    return 'plugin result'\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "api_keys.json"
    config_path.write_text(json.dumps({
        "plugins_enabled": {"sample_plugin": True},
        "plugin_config": {"sample_plugin": {"token": "test-value"}},
    }), encoding="utf-8")
    monkeypatch.setattr(config_manager, "CONFIG_FILE", config_path)

    registry = plugin_loader.discover_plugins(
        plugin_dir, core_tool_names=set(), logger=lambda _line: None)

    assert registry.has("sample_plugin")
    assert registry.run("sample_plugin", {}) == "plugin result"
    assert config_manager.get_plugin_config("sample_plugin") == {"token": "test-value"}
    assert registry.get_tool_declarations()[0]["name"] == "sample_plugin"


def test_computer_settings_uses_only_ui_context_to_bypass_its_inner_gate(monkeypatch):
    from actions import computer_settings

    calls, prompts = [], []
    monkeypatch.setattr(computer_settings, "_PYAUTOGUI", True)
    monkeypatch.setitem(computer_settings.ACTION_MAP, "shutdown",
                        lambda: calls.append("shutdown"))
    monkeypatch.setattr(computer_settings.confirm, "pending_title", lambda: "")
    monkeypatch.setattr(computer_settings.confirm, "request",
                        lambda **kwargs: prompts.append(kwargs) or "[CONFIRMATION_PENDING]")

    # A model-controlled parameter with the private name is not a UI approval.
    pending = computer_settings.computer_settings(
        {"action": "shutdown", "_confirmation_approved": True})
    assert pending == "[CONFIRMATION_PENDING]"
    assert not calls and len(prompts) == 1

    # Only the trusted registry context from task_agent's accepted UI callback
    # bypasses the action's inner gate, avoiding a second banner.
    result = computer_settings.computer_settings(
        {"action": "shutdown"}, _confirmation_approved=True)
    assert result == "shutdown done."
    assert calls == ["shutdown"]


def test_new_action_and_job_modules_remain_discoverable():
    root = Path(__file__).resolve().parent.parent
    registry = action_loader.discover_actions(
        root / "actions", reserved_names=set(), logger=lambda _line: None)

    assert registry.has("task_agent")
    assert registry.has("jobs")
    assert registry.has("research")
    # Discovery populates the same shared registry API used by task execution.
    assert action_loader.get_registry() is registry
