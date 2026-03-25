import os
from pathlib import Path
from types import SimpleNamespace

from drive_vertex_cli.auth_setup import AuthSetupSummary
from drive_vertex_cli.client import (
    DriveVertexChatSession,
    DriveVertexClient,
    DriveVertexStatus,
)
from drive_vertex_cli.index_store import IndexManifest
from drive_vertex_cli.indexer import SyncStats
from drive_vertex_cli.retrieval import RetrievalAnswer
from drive_vertex_cli.env_file import read_env_values


def _settings(**overrides):
    values = {
        "google_cloud_project": "test-project",
        "google_cloud_location": "us-central1",
        "google_application_credentials": None,
        "gemini_model": "gemini-2.5-flash",
        "gemini_temperature": 0.2,
        "embedding_model": "text-embedding-005",
        "embedding_dimensions": 768,
        "default_top_k": 5,
        "conversation_max_turns": 6,
        "drive_service_account_file": None,
        "drive_oauth_client_secret_file": None,
        "drive_token_file": Path(".secrets/google-drive-token.json"),
        "index_dir": Path(".cache/drive-vertex-index"),
        "default_folder_id": "folder-123",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_client_sync_uses_default_folder_and_index_dir(monkeypatch):
    client = DriveVertexClient(_settings())
    captured: dict[str, object] = {}

    def fake_sync_folder(**kwargs):
        captured.update(kwargs)
        return SyncStats(
            indexed_file_count=1,
            skipped_file_count=0,
            chunk_count=3,
            skipped_reasons=[],
        )

    monkeypatch.setattr("drive_vertex_cli.client.sync_folder", fake_sync_folder)

    result = client.sync()

    assert result.indexed_file_count == 1
    assert captured["folder_id"] == "folder-123"
    assert captured["index_dir"] == Path(".cache/drive-vertex-index")


def test_client_sync_interactive_chooses_folder_when_missing(monkeypatch):
    client = DriveVertexClient(_settings(default_folder_id=None))
    captured: dict[str, object] = {}

    def fake_sync_folder(**kwargs):
        captured.update(kwargs)
        return SyncStats(
            indexed_file_count=1,
            skipped_file_count=0,
            chunk_count=2,
            skipped_reasons=[],
        )

    monkeypatch.setattr("drive_vertex_cli.client.sync_folder", fake_sync_folder)
    monkeypatch.setattr(
        client,
        "choose_folder",
        lambda **kwargs: SimpleNamespace(folder_id="picked-folder"),
    )

    result = client.sync(interactive=True)

    assert result.chunk_count == 2
    assert captured["folder_id"] == "picked-folder"


def test_setup_env_updates_process_environment_and_writes_env_file(tmp_path: Path):
    env_file = tmp_path / ".env"
    keys = [
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "GOOGLE_DRIVE_OAUTH_CLIENT_SECRET_FILE",
        "GOOGLE_DRIVE_TOKEN_FILE",
        "GOOGLE_DRIVE_FOLDER_ID",
        "DRIVE_VERTEX_INDEX_DIR",
        "VERTEX_GEMINI_MODEL",
        "VERTEX_GEMINI_TEMPERATURE",
        "VERTEX_EMBEDDING_MODEL",
        "VERTEX_EMBEDDING_DIMENSIONS",
        "DRIVE_VERTEX_DEFAULT_TOP_K",
        "DRIVE_VERTEX_CONVERSATION_MAX_TURNS",
    ]
    previous = {key: os.environ.get(key) for key in keys}
    try:
        updates = DriveVertexClient.setup_env(
            google_cloud_project="project-42439",
            drive_oauth_client_secret_file=".secrets/google-drive-oauth-client.json",
            drive_token_file=".secrets/google-drive-token.json",
            default_folder_id="folder-123",
            index_dir=".cache/drive-vertex-index",
            env_file=env_file,
        )

        assert updates["GOOGLE_CLOUD_PROJECT"] == "project-42439"
        assert updates["GOOGLE_DRIVE_FOLDER_ID"] == "folder-123"
        assert updates["VERTEX_GEMINI_MODEL"] == "gemini-2.5-flash"
        assert os.environ["GOOGLE_CLOUD_PROJECT"] == "project-42439"
        assert os.environ["GOOGLE_DRIVE_FOLDER_ID"] == "folder-123"

        values = read_env_values(env_file)
        assert values["GOOGLE_CLOUD_PROJECT"] == "project-42439"
        assert values["GOOGLE_DRIVE_OAUTH_CLIENT_SECRET_FILE"] == ".secrets/google-drive-oauth-client.json"
        assert values["GOOGLE_DRIVE_TOKEN_FILE"] == ".secrets/google-drive-token.json"
        assert values["GOOGLE_DRIVE_FOLDER_ID"] == "folder-123"
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_setup_env_interactive_reuses_auth_cli_and_loads_written_values(tmp_path: Path):
    env_file = tmp_path / ".env"
    keys = [
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_DRIVE_OAUTH_CLIENT_SECRET_FILE",
        "GOOGLE_DRIVE_TOKEN_FILE",
    ]
    previous = {key: os.environ.get(key) for key in keys}

    def fake_run_auth_cli(path: Path) -> None:
        path.write_text(
            "GOOGLE_CLOUD_PROJECT=project-42439\n"
            "GOOGLE_DRIVE_OAUTH_CLIENT_SECRET_FILE=.secrets/google-drive-oauth-client.json\n"
            "GOOGLE_DRIVE_TOKEN_FILE=.secrets/google-drive-token.json\n"
        )

    try:
        from drive_vertex_cli import client as client_module

        original = client_module._run_auth_cli
        client_module._run_auth_cli = fake_run_auth_cli
        try:
            updates = DriveVertexClient.setup_env(interactive=True, env_file=env_file)
        finally:
            client_module._run_auth_cli = original

        assert updates["GOOGLE_CLOUD_PROJECT"] == "project-42439"
        assert os.environ["GOOGLE_CLOUD_PROJECT"] == "project-42439"
        assert (
            os.environ["GOOGLE_DRIVE_OAUTH_CLIENT_SECRET_FILE"]
            == ".secrets/google-drive-oauth-client.json"
        )
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_get_folders_returns_serializable_folder_mappings(monkeypatch):
    client = DriveVertexClient(_settings())
    monkeypatch.setattr(
        client,
        "list_folders",
        lambda: [
            SimpleNamespace(folder_id="folder-1", name="Alpha", web_view_link=None),
            SimpleNamespace(
                folder_id="folder-2",
                name="Beta",
                web_view_link="https://drive.google.com/drive/folders/folder-2",
            ),
        ],
    )

    folders = client.get_folders()

    assert folders == [
        {
            "folder_id": "folder-1",
            "name": "Alpha",
            "web_view_link": None,
        },
        {
            "folder_id": "folder-2",
            "name": "Beta",
            "web_view_link": "https://drive.google.com/drive/folders/folder-2",
        },
    ]


def test_client_setup_auth_delegates_to_auth_setup(monkeypatch):
    client = DriveVertexClient(_settings())
    captured: dict[str, object] = {}

    def fake_complete_auth_setup(*, settings, vertex_auth_mode, drive_auth_mode):
        captured["settings"] = settings
        captured["vertex_auth_mode"] = vertex_auth_mode
        captured["drive_auth_mode"] = drive_auth_mode
        return AuthSetupSummary(completed_steps=["ok"])

    monkeypatch.setattr(
        "drive_vertex_cli.client.complete_auth_setup",
        fake_complete_auth_setup,
    )

    summary = client.setup_auth(
        vertex_auth_mode="service-account",
        drive_auth_mode="oauth",
    )

    assert summary.completed_steps == ["ok"]
    assert captured["settings"] is client.settings
    assert captured["vertex_auth_mode"] == "service-account"
    assert captured["drive_auth_mode"] == "oauth"


def test_choose_folder_updates_settings_and_env(monkeypatch, tmp_path: Path):
    client = DriveVertexClient(_settings(default_folder_id=None))
    previous = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")

    monkeypatch.setattr("drive_vertex_cli.client.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(
        client,
        "list_folders",
        lambda: [
            SimpleNamespace(folder_id="folder-1", name="Alpha", web_view_link=None),
            SimpleNamespace(folder_id="folder-2", name="Beta", web_view_link=None),
        ],
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": "2")

    try:
        selected = client.choose_folder(env_file=tmp_path / ".env")
        assert selected.folder_id == "folder-2"
        assert client.settings.default_folder_id == "folder-2"
        assert os.environ["GOOGLE_DRIVE_FOLDER_ID"] == "folder-2"
        assert (tmp_path / ".env").read_text().strip() == "GOOGLE_DRIVE_FOLDER_ID=folder-2"
    finally:
        if previous is None:
            os.environ.pop("GOOGLE_DRIVE_FOLDER_ID", None)
        else:
            os.environ["GOOGLE_DRIVE_FOLDER_ID"] = previous


def test_client_auth_alias_calls_setup_auth(monkeypatch):
    client = DriveVertexClient(_settings())
    captured: dict[str, object] = {}

    def fake_setup_auth(*, vertex_auth_mode, drive_auth_mode):
        captured["vertex_auth_mode"] = vertex_auth_mode
        captured["drive_auth_mode"] = drive_auth_mode
        return AuthSetupSummary(completed_steps=["ok"])

    monkeypatch.setattr(client, "setup_auth", fake_setup_auth)

    summary = client.auth(
        vertex_auth_mode="adc",
        drive_auth_mode="service-account",
    )

    assert summary.completed_steps == ["ok"]
    assert captured["vertex_auth_mode"] == "adc"
    assert captured["drive_auth_mode"] == "service-account"


def test_client_ask_uses_settings_defaults(monkeypatch):
    client = DriveVertexClient(_settings())
    captured: dict[str, object] = {}

    class FakeRetriever:
        def answer(self, question, **kwargs):
            captured["question"] = question
            captured.update(kwargs)
            return RetrievalAnswer(answer="ok", hits=[])

    monkeypatch.setattr(client, "build_retriever", lambda **kwargs: FakeRetriever())

    result = client.ask("What does SQG do?")

    assert result.answer == "ok"
    assert captured["question"] == "What does SQG do?"
    assert captured["model"] == "gemini-2.5-flash"
    assert captured["default_top_k"] == 5
    assert captured["temperature"] == 0.2
    assert captured["conversation_max_turns"] == 6


def test_chat_session_tracks_history():
    client = DriveVertexClient(_settings())
    calls: list[list[tuple[str, str]] | None] = []
    answers = iter(["first answer", "second answer"])

    client.ask = lambda question, **kwargs: (  # type: ignore[method-assign]
        calls.append(kwargs.get("conversation_history"))
        or RetrievalAnswer(answer=next(answers), hits=[])
    )

    session = DriveVertexChatSession(client=client)
    first = session.ask("First?")
    second = session.ask("Second?")

    assert first.answer == "first answer"
    assert second.answer == "second answer"
    assert calls == [[], [("First?", "first answer")]]


def test_status_returns_library_friendly_metadata(monkeypatch):
    client = DriveVertexClient(_settings())
    manifest = IndexManifest(
        version=1,
        synced_at="2026-03-25T00:00:00+00:00",
        folder_id="folder-123",
        embedding_model="text-embedding-005",
        embedding_dimensions=768,
        chunk_size_tokens=350,
        chunk_overlap_tokens=60,
        file_count=2,
        chunk_count=7,
    )
    fake_index = SimpleNamespace(manifest=manifest)
    monkeypatch.setattr("drive_vertex_cli.client.LocalIndex.load", lambda index_dir: fake_index)

    status = client.status()

    assert isinstance(status, DriveVertexStatus)
    assert status.index_dir == Path(".cache/drive-vertex-index")
    assert status.manifest.folder_id == "folder-123"
    assert status.gemini_model == "gemini-2.5-flash"
    assert status.default_top_k == 5
