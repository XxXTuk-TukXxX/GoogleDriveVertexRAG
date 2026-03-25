from pathlib import Path
from types import SimpleNamespace

from drive_vertex_cli.cli import _resolve_sync_folder_id
from drive_vertex_cli.drive_client import DriveFolderOption


def test_resolve_sync_folder_id_interactively_saves_selection(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)

    prompts = iter(["2"])
    monkeypatch.setattr("drive_vertex_cli.cli.typer.prompt", lambda *args, **kwargs: next(prompts))
    monkeypatch.setattr("drive_vertex_cli.cli.typer.echo", lambda *args, **kwargs: None)
    monkeypatch.setattr("drive_vertex_cli.cli.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("drive_vertex_cli.cli.build_drive_service", lambda settings: object())
    monkeypatch.setattr(
        "drive_vertex_cli.cli.list_accessible_folders",
        lambda service: [
            DriveFolderOption(folder_id="folder-1", name="Alpha", web_view_link=None),
            DriveFolderOption(folder_id="folder-2", name="Beta", web_view_link=None),
        ],
    )

    settings = SimpleNamespace(default_folder_id=None)

    selected = _resolve_sync_folder_id(None, settings)

    assert selected == "folder-2"
    assert (tmp_path / ".env").read_text().strip() == "GOOGLE_DRIVE_FOLDER_ID=folder-2"


def test_resolve_sync_folder_id_uses_existing_default_when_not_interactive(monkeypatch):
    monkeypatch.setattr("drive_vertex_cli.cli.sys.stdin.isatty", lambda: False)
    settings = SimpleNamespace(default_folder_id="saved-folder")

    assert _resolve_sync_folder_id(None, settings) == "saved-folder"
