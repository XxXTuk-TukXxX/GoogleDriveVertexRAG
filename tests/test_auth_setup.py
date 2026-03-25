import os
import subprocess
from pathlib import Path

import pytest

from drive_vertex_cli.auth_setup import (
    AuthSetupError,
    _run_command,
    apply_env_updates,
    validate_existing_file,
)


def test_apply_env_updates_sets_and_clears_values(monkeypatch):
    monkeypatch.setenv("KEEP_ME", "value")
    monkeypatch.setenv("REMOVE_ME", "gone")

    apply_env_updates({"KEEP_ME": "new", "REMOVE_ME": "", "ADD_ME": "hello"})

    assert os.environ["KEEP_ME"] == "new"
    assert "REMOVE_ME" not in os.environ
    assert os.environ["ADD_ME"] == "hello"


def test_validate_existing_file_returns_expanded_path(tmp_path: Path):
    file_path = tmp_path / "client.json"
    file_path.write_text("{}")

    resolved = validate_existing_file(str(file_path), label="OAuth client file")

    assert resolved == file_path


def test_validate_existing_file_raises_for_missing_file(tmp_path: Path):
    with pytest.raises(AuthSetupError):
        validate_existing_file(str(tmp_path / "missing.json"), label="OAuth client file")


def test_run_command_strips_empty_google_credential_env_vars(monkeypatch):
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    monkeypatch.setenv("GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE", "")
    monkeypatch.setenv("KEEP_ME", "value")

    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("drive_vertex_cli.auth_setup.subprocess.run", fake_run)

    result = _run_command(["gcloud", "--version"], capture_output=True)

    assert result.stdout == "ok"
    env = captured["env"]
    assert isinstance(env, dict)
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in env
    assert "GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE" not in env
    assert env["KEEP_ME"] == "value"
