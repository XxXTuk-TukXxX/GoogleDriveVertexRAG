from pathlib import Path
from types import SimpleNamespace

from drive_vertex_cli.client import (
    DriveVertexChatSession,
    DriveVertexClient,
    DriveVertexStatus,
)
from drive_vertex_cli.index_store import IndexManifest
from drive_vertex_cli.indexer import SyncStats
from drive_vertex_cli.retrieval import RetrievalAnswer


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
