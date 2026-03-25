# Library Reference

`drive_vertex_cli` is structured as a small library first, with the `drive-vertex` CLI layered on top of the same public API.

## Install From GitHub

```bash
pip install "git+https://github.com/XxXTuk-TukXxX/GoogleDriveVertexRAG.git"
```

For editable local development:

```bash
git clone https://github.com/XxXTuk-TukXxX/GoogleDriveVertexRAG.git
cd GoogleDriveVertexRAG
pip install -e ".[dev]"
```

## Main Entry Point

Use `DriveVertexClient` for most integrations:

```python
from drive_vertex_cli import DriveVertexClient

client = DriveVertexClient.from_env()
```

### `DriveVertexClient.from_env()`

Builds a client from the current process environment. This is the standard entry point for apps that keep configuration in `.env` or environment variables.

### `DriveVertexClient.setup_env()`

Populates the process environment for later `DriveVertexClient.from_env()` calls. You can also pass `env_file=...` to write the same values to a dotenv file.

```python
from drive_vertex_cli import DriveVertexClient

DriveVertexClient.setup_env(
    google_cloud_project="project-42439",
    drive_oauth_client_secret_file=".secrets/google-drive-oauth-client.json",
    drive_token_file=".secrets/google-drive-token.json",
    default_folder_id="YOUR_FOLDER_ID",
    env_file=".env",
)

client = DriveVertexClient.from_env()
```

This is useful when you want to set the library up programmatically instead of manually editing `.env` first.

If you want the same interactive experience as the CLI `auth` command, use:

```python
from drive_vertex_cli import DriveVertexClient

DriveVertexClient.setup_env(interactive=True, env_file=".env")
client = DriveVertexClient.from_env()
```

That path launches the existing `drive-vertex auth` flow and then reloads the written env values into the current Python process.

### `DriveVertexClient.auth()`

Creates or verifies local auth state, enables the required Google APIs, and verifies Drive access using the same underlying setup logic as the CLI `auth` command.

```python
from drive_vertex_cli import DriveVertexClient

client = DriveVertexClient.from_env()
summary = client.auth(
    vertex_auth_mode="adc",
    drive_auth_mode="oauth",
)

print(summary.completed_steps)
print(summary.warnings)
```

This method assumes the environment variables are already configured. It does not prompt for credential file paths or write `.env` for you.

`DriveVertexClient.setup_auth()` is still available as the more explicit alias.

### `DriveVertexClient.sync()`

Rebuilds the local vector index from a Google Drive folder.

```python
stats = client.sync(folder_id="YOUR_FOLDER_ID")
print(stats.indexed_file_count)
```

If `GOOGLE_DRIVE_FOLDER_ID` is already configured, you can omit `folder_id`.

If you want the library to prompt you to choose a folder when none is configured, use:

```python
stats = client.sync(interactive=True)
```

### `DriveVertexClient.choose_folder()`

Shows the visible Google Drive folders in the terminal, lets you pick one, and saves the selected folder as `GOOGLE_DRIVE_FOLDER_ID` for later runs.

```python
selected = client.choose_folder()
print(selected.folder_id, selected.name)
```

### `DriveVertexClient.get_folders()`

Returns the visible Google Drive folders as plain dictionaries so they are easy to serialize, log, or store.

```python
folders = client.get_folders()
print(folders[0]["name"])
```

### `DriveVertexClient.ask()`

Runs one retrieval-augmented question against the local index.

```python
result = client.ask("What does the SQG project do?")
print(result.answer)
```

### `DriveVertexClient.open_chat()`

Creates a reusable chat session that keeps previous turns as context.

```python
chat = client.open_chat()
reply = chat.ask("Summarize the onboarding notes.")
```

### `DriveVertexClient.status()`

Returns library-friendly metadata about the current local index and active AI settings.

## Public Types

- `AuthSetupSummary`: outcome of a programmatic auth bootstrap run
- `DriveVertexClient`: high-level sync, ask, chat, and status facade
- `DriveVertexChatSession`: stateful conversation helper
- `DriveVertexStatus`: resolved index/settings snapshot
- `RetrievalAnswer`: answer text plus supporting search hits
- `SyncStats`: sync counters and skipped-file reasons
- `LocalIndex`: low-level local vector index object

## Module Guide

- `client.py`: public library facade
- `config.py`: environment-backed settings loader
- `drive_client.py`: Google Drive authentication and file traversal
- `indexer.py`: Drive-to-index pipeline
- `retrieval.py`: Gemini retrieval flow and prompt assembly
- `vertex_client.py`: Vertex AI SDK wrapper

## Important Runtime Notes

- `setup_env()` prepares the environment for `from_env()`. In programmatic mode it sets env values directly; in `interactive=True` mode it launches the full CLI auth flow and reloads the resulting dotenv values.
- `auth()` performs the real credential/API verification work, but it does not collect user input or write `.env`.
- Changing embedding settings only affects future `sync()` runs.
- `ask()` and chat read from the local index on disk; they do not query Drive directly.
- For personal Google Drive access, OAuth is typically the correct auth mode.
- For unattended usage, a service account can be used if the Drive folder is shared with it.
