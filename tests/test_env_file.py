from pathlib import Path

from drive_vertex_cli.env_file import read_env_values, serialize_env_value, upsert_env_file


def test_upsert_env_file_updates_existing_keys_and_preserves_other_lines(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\nGOOGLE_CLOUD_PROJECT=old-project\nOTHER_KEY=value\n")

    upsert_env_file(
        env_file,
        {
            "GOOGLE_CLOUD_PROJECT": "new-project",
            "GOOGLE_DRIVE_FOLDER_ID": "folder-123",
        },
    )

    content = env_file.read_text()
    assert "# comment" in content
    assert "GOOGLE_CLOUD_PROJECT=new-project" in content
    assert "OTHER_KEY=value" in content
    assert "GOOGLE_DRIVE_FOLDER_ID=folder-123" in content


def test_serialize_env_value_quotes_paths_with_spaces():
    assert serialize_env_value("/tmp/no-spaces.json") == "/tmp/no-spaces.json"
    assert serialize_env_value("/tmp/with space.json") == '"/tmp/with space.json"'


def test_read_env_values_returns_plain_mapping(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("GOOGLE_CLOUD_PROJECT=test-project\nEMPTY=\n")

    values = read_env_values(env_file)

    assert values["GOOGLE_CLOUD_PROJECT"] == "test-project"
    assert values["EMPTY"] == ""
