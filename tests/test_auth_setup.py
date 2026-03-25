import os
from pathlib import Path

import pytest

from drive_vertex_cli.auth_setup import AuthSetupError, apply_env_updates, validate_existing_file


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
