"""Guard the real startup/shutdown seam without importing Qt or audio devices."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _function_source(name):
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    return ""


def test_worker_is_started_once_in_the_real_live_lifecycle():
    body = _function_source("run")

    assert "start_worker()" in body
    assert "reconnect" in body.lower()
    # Startup is outside the reconnect loop, so reconnecting Gemini Live does
    # not create duplicate schedulers.
    assert body.index("start_worker()") < body.index("while True:")


def test_worker_is_stopped_on_ui_exit_and_explicit_assistant_shutdown():
    main_body = _function_source("main")
    execute_tool = _function_source("_execute_tool")

    assert "aboutToQuit.connect" in main_body
    assert "stop_worker" in main_body
    assert "stop_worker" in execute_tool


def test_job_controls_are_exposed_as_an_existing_discoverable_action():
    source = (ROOT / "actions" / "jobs.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    tool = next(node.value for node in tree.body
                if isinstance(node, ast.Assign)
                and any(getattr(target, "id", None) == "TOOL" for target in node.targets))
    assert "jobs" in ast.unparse(tool)
    for action in ("submit", "status", "list", "cancel", "resume", "logs", "report"):
        assert action in source
