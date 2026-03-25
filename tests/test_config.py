import pytest

from drive_vertex_cli.config import ConfigurationError, load_settings


def test_load_settings_reads_ai_overrides(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("VERTEX_GEMINI_MODEL", "gemini-2.5-pro")
    monkeypatch.setenv("VERTEX_GEMINI_TEMPERATURE", "0.7")
    monkeypatch.setenv("VERTEX_EMBEDDING_MODEL", "gemini-embedding-001")
    monkeypatch.setenv("VERTEX_EMBEDDING_DIMENSIONS", "1536")
    monkeypatch.setenv("DRIVE_VERTEX_DEFAULT_TOP_K", "8")
    monkeypatch.setenv("DRIVE_VERTEX_CONVERSATION_MAX_TURNS", "10")

    settings = load_settings()

    assert settings.gemini_model == "gemini-2.5-pro"
    assert settings.gemini_temperature == 0.7
    assert settings.embedding_model == "gemini-embedding-001"
    assert settings.embedding_dimensions == 1536
    assert settings.default_top_k == 8
    assert settings.conversation_max_turns == 10


def test_load_settings_rejects_invalid_ai_numeric_values(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("VERTEX_GEMINI_TEMPERATURE", "not-a-number")

    with pytest.raises(ConfigurationError):
        load_settings()
