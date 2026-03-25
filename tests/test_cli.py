import os
from pathlib import Path

from typer.testing import CliRunner

from drive_vertex_cli.cli import AI_DEFAULTS
from drive_vertex_cli.cli import _effective_ai_values
from drive_vertex_cli.cli import _missing_file_guidance
from drive_vertex_cli.cli import app
from drive_vertex_cli.env_file import read_env_values


def test_missing_file_guidance_for_drive_oauth_client_includes_links():
    lines = _missing_file_guidance(
        kind="drive_oauth_client",
        project_id="project-42439",
        target_path=".secrets/client.json",
    )

    message = "\n".join(lines)
    assert "console.cloud.google.com/apis/library/drive.googleapis.com?project=project-42439" in message
    assert "console.cloud.google.com/auth/audience?project=project-42439" in message
    assert "Access blocked" in message
    assert ".secrets/client.json" in message
    assert "Docs:" not in message


def test_missing_file_guidance_for_drive_service_account_mentions_folder_share():
    lines = _missing_file_guidance(
        kind="drive_service_account",
        project_id="project-42439",
        target_path=".secrets/drive-service-account.json",
    )

    message = "\n".join(lines)
    assert "console.cloud.google.com/iam-admin/serviceaccounts?project=project-42439" in message
    assert "Share the target Google Drive folder with the service account email" in message
    assert "Docs:" not in message


def test_effective_ai_values_fill_in_defaults():
    values = _effective_ai_values({"VERTEX_GEMINI_MODEL": "gemini-2.5-pro"})

    assert values["VERTEX_GEMINI_MODEL"] == "gemini-2.5-pro"
    assert values["VERTEX_GEMINI_TEMPERATURE"] == AI_DEFAULTS["VERTEX_GEMINI_TEMPERATURE"]
    assert values["DRIVE_VERTEX_DEFAULT_TOP_K"] == AI_DEFAULTS["DRIVE_VERTEX_DEFAULT_TOP_K"]


def test_ai_command_updates_env_file(tmp_path: Path):
    env_file = tmp_path / ".env"
    runner = CliRunner()

    previous = {key: os.environ.get(key) for key in AI_DEFAULTS}
    try:
        result = runner.invoke(
            app,
            [
                "ai",
                "--env-file",
                str(env_file),
                "--gemini-model",
                "gemini-2.5-pro",
                "--embedding-model",
                "gemini-embedding-001",
                "--embedding-dimensions",
                "auto",
                "--temperature",
                "0.7",
                "--default-top-k",
                "8",
                "--conversation-max-turns",
                "10",
            ],
        )
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    assert result.exit_code == 0, result.stdout

    values = read_env_values(env_file)
    assert values["VERTEX_GEMINI_MODEL"] == "gemini-2.5-pro"
    assert values["VERTEX_EMBEDDING_MODEL"] == "gemini-embedding-001"
    assert values["VERTEX_EMBEDDING_DIMENSIONS"] == ""
    assert values["VERTEX_GEMINI_TEMPERATURE"] == "0.7"
    assert values["DRIVE_VERTEX_DEFAULT_TOP_K"] == "8"
    assert values["DRIVE_VERTEX_CONVERSATION_MAX_TURNS"] == "10"
