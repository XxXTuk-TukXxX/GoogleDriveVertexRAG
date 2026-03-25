# Drive Vertex

`drive-vertex` is a Python library with a CLI on top. It:

1. Reads files from a Google Drive folder.
2. Extracts text from supported file types.
3. Chunks and embeds the corpus with Vertex AI.
4. Stores a local vector index on disk.
5. Lets Gemini call a retrieval tool over that index to answer questions.

The implementation is intentionally simple: Google Drive is the system of record, Vertex AI provides embeddings and generation, and the vector index lives locally in `.cache/drive-vertex-index` by default.

## Supported File Types

- Google Docs
- Google Slides
- Google Sheets
- Plain text, Markdown, JSON, CSV, HTML, XML
- PDF
- DOCX
- PPTX
- XLSX

## Prerequisites

- Python 3.11+
- A Google Cloud project with:
  - Vertex AI API enabled
  - Google Drive API enabled
- Vertex AI auth configured through Application Default Credentials:
  - `gcloud auth application-default login`, or
  - `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json`
- Google Drive auth configured with one of:
  - `GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE=/path/to/service-account.json`
  - `GOOGLE_DRIVE_OAUTH_CLIENT_SECRET_FILE=/path/to/oauth-client-secret.json`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
drive-vertex auth
```

If you prefer to edit the file manually, you can still `cp .env.example .env` and fill it in yourself.

`drive-vertex auth` now does more than write `.env`:

- validates any credential file paths you enter
- prints console steps when a required credential file is missing
- creates or verifies Vertex AI Application Default Credentials when you choose `adc`
- enables `aiplatform.googleapis.com` and `drive.googleapis.com` on the selected project
- creates or verifies the Google Drive OAuth token when you choose `oauth`
- verifies the configured Drive folder is reachable

Show or update the AI settings used by `ask` and future `sync` runs:

```bash
drive-vertex ai
drive-vertex ai --show
drive-vertex ai --gemini-model gemini-2.5-pro --temperature 0.4
```

The AI settings command controls:

- Gemini model
- Gemini temperature
- Embedding model
- Embedding dimensions
- Default retrieval `top-k`
- Conversation history turns kept in interactive chat

If you use a Drive service account, share the source Drive folder with that service account email. If you use the desktop OAuth client flow, the CLI will open a browser during `auth` when it needs to create the Drive token and will cache the token in `.secrets/google-drive-token.json`.

If your Google Auth Platform app is `External` and not yet verified, add yourself as a test user on the Audience page before trying Drive OAuth, otherwise you can hit an error like `Access blocked: <project> has not completed the Google verification process`. The project-scoped Audience page is:

```text
https://console.cloud.google.com/auth/audience?project=YOUR_PROJECT_ID
```

## Library Usage

Import the library directly when you want to build this into another Python app:

```python
from drive_vertex_cli import DriveVertexClient

client = DriveVertexClient.from_env()
client.sync(folder_id="YOUR_FOLDER_ID")

answer = client.ask("What does the SQG project do?")
print(answer.answer)

chat = client.open_chat()
reply = chat.ask("Summarize the latest onboarding notes.")
print(reply.answer)
```

Useful public types exported by the package include:

- `DriveVertexClient`
- `DriveVertexChatSession`
- `DriveVertexStatus`
- `RetrievalAnswer`
- `SyncStats`
- `LocalIndex`

## CLI Usage

Create or update `.env` interactively:

```bash
drive-vertex auth
```

Sync a Drive folder into the local vector index:

```bash
drive-vertex sync
```

If you do not pass `--folder-id`, the CLI lists the Google Drive folders visible to the authenticated account, lets you choose one by number, and saves the selected folder ID into `.env` as the new default.

Ask a question against the indexed corpus:

```bash
drive-vertex ask "What does our onboarding doc say about access reviews?"
```

Start an interactive chat session:

```bash
drive-vertex ask
```

Rebuild the index and then answer:

```bash
drive-vertex ask "Summarize the QBR deck" --refresh --folder-id YOUR_FOLDER_ID
```

See current index metadata:

```bash
drive-vertex status
```

You can also inspect the currently configured AI settings without editing them:

```bash
drive-vertex ai --show
```

## How It Works

- `sync` walks the target Drive folder recursively.
- Each supported file is downloaded or exported, converted into plain text, tokenized, and chunked.
- Chunks are embedded with Vertex AI using `RETRIEVAL_DOCUMENT`.
- The CLI saves:
  - `manifest.json`
  - `chunks.jsonl`
  - `embeddings.npy`
- `ask` loads the local index, embeds the user query with `RETRIEVAL_QUERY`, and exposes a `search_drive_corpus` tool to Gemini.
- Gemini calls the tool, receives the most relevant snippets, and produces the final answer with cited source file names.
- Changing the embedding model or embedding dimensions only affects future `sync` runs. Run `drive-vertex sync` again after changing them.

## Notes

- This is a full rebuild indexer, not an incremental sync engine.
- The local vector search uses normalized cosine similarity in NumPy.
- Large corpora may take time to embed because the default embedding API has request limits. The CLI batches requests where the model allows it.
