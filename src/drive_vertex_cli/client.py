from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from drive_vertex_cli.auth_setup import AuthSetupSummary, apply_env_updates, complete_auth_setup
from drive_vertex_cli.config import ConfigurationError, Settings, load_settings
from drive_vertex_cli.drive_client import DriveFolderOption, build_drive_service, list_accessible_folders
from drive_vertex_cli.env_file import read_env_values, upsert_env_file
from drive_vertex_cli.index_store import IndexManifest, LocalIndex
from drive_vertex_cli.indexer import SyncStats, sync_folder
from drive_vertex_cli.retrieval import DriveCorpusRetriever, RetrievalAnswer
from drive_vertex_cli.vertex_client import VertexClient

DEFAULT_ENV_FILE = Path(".env")
CONSOLE = Console()
ERROR_CONSOLE = Console(stderr=True)


@dataclass(slots=True)
class DriveVertexStatus:
    """Resolved status view for the current local index and runtime settings."""

    index_dir: Path
    manifest: IndexManifest
    gemini_model: str
    gemini_temperature: float
    configured_embedding_model: str
    configured_embedding_dimensions: int | None
    default_top_k: int
    conversation_max_turns: int


@dataclass(slots=True)
class DriveVertexChatSession:
    """Stateful chat helper that keeps conversation history between questions."""

    client: "DriveVertexClient"
    index_dir: Path | None = None
    top_k: int | None = None
    model: str | None = None
    temperature: float | None = None
    conversation_max_turns: int | None = None
    history: list[tuple[str, str]] = field(default_factory=list)

    def ask(self, question: str, *, top_k: int | None = None) -> RetrievalAnswer:
        """Ask a follow-up question while preserving earlier turns as context."""

        result = self.client.ask(
            question,
            top_k=top_k if top_k is not None else self.top_k,
            index_dir=self.index_dir,
            model=self.model,
            temperature=self.temperature,
            conversation_max_turns=self.conversation_max_turns,
            conversation_history=list(self.history),
        )
        self.history.append((question, result.answer))
        return result

    def clear(self) -> None:
        """Discard any stored conversation history for the session."""

        self.history.clear()


class DriveVertexClient:
    """High-level library facade for syncing, retrieval, and chat workflows."""

    def __init__(
        self,
        settings: Settings,
        *,
        vertex: VertexClient | None = None,
    ) -> None:
        self.settings = settings
        self._vertex = vertex

    @classmethod
    def from_env(
        cls: type["DriveVertexClient"],
        *,
        require_project: bool = True,
    ) -> "DriveVertexClient":
        """Build a client from the current process environment variables."""

        return cls(load_settings(require_project=require_project))

    @staticmethod
    def setup_env(
        *,
        google_cloud_project: str = "",
        google_cloud_location: str = "us-central1",
        google_application_credentials: Path | str | None = None,
        drive_service_account_file: Path | str | None = None,
        drive_oauth_client_secret_file: Path | str | None = None,
        drive_token_file: Path | str | None = None,
        default_folder_id: str | None = None,
        index_dir: Path | str | None = None,
        gemini_model: str = "gemini-2.5-flash",
        gemini_temperature: float = 0.2,
        embedding_model: str = "text-embedding-005",
        embedding_dimensions: int | None = 768,
        default_top_k: int = 5,
        conversation_max_turns: int = 6,
        env_file: Path | str | None = None,
        interactive: bool = False,
    ) -> dict[str, str]:
        """Populate environment variables for later `from_env()` calls.

        When `env_file` is provided, the same values are also written to that dotenv file.
        """

        if interactive:
            env_path = Path(env_file) if env_file is not None else DEFAULT_ENV_FILE
            _run_auth_cli(env_path)
            updates = read_env_values(env_path)
            apply_env_updates(updates)
            return updates

        if not google_cloud_project.strip():
            raise ValueError("google_cloud_project is required.")
        if gemini_temperature < 0:
            raise ValueError("gemini_temperature must be at least 0.")
        if embedding_dimensions is not None and embedding_dimensions <= 0:
            raise ValueError("embedding_dimensions must be greater than 0 when provided.")
        if default_top_k < 1:
            raise ValueError("default_top_k must be at least 1.")
        if conversation_max_turns < 0:
            raise ValueError("conversation_max_turns must be at least 0.")

        updates = {
            "GOOGLE_CLOUD_PROJECT": google_cloud_project.strip(),
            "GOOGLE_CLOUD_LOCATION": google_cloud_location,
            "GOOGLE_APPLICATION_CREDENTIALS": _stringify_env_value(
                google_application_credentials
            ),
            "GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE": _stringify_env_value(
                drive_service_account_file
            ),
            "GOOGLE_DRIVE_OAUTH_CLIENT_SECRET_FILE": _stringify_env_value(
                drive_oauth_client_secret_file
            ),
            "GOOGLE_DRIVE_TOKEN_FILE": _stringify_env_value(drive_token_file),
            "GOOGLE_DRIVE_FOLDER_ID": default_folder_id or "",
            "DRIVE_VERTEX_INDEX_DIR": _stringify_env_value(index_dir),
            "VERTEX_GEMINI_MODEL": gemini_model,
            "VERTEX_GEMINI_TEMPERATURE": str(gemini_temperature),
            "VERTEX_EMBEDDING_MODEL": embedding_model,
            "VERTEX_EMBEDDING_DIMENSIONS": (
                "" if embedding_dimensions is None else str(embedding_dimensions)
            ),
            "DRIVE_VERTEX_DEFAULT_TOP_K": str(default_top_k),
            "DRIVE_VERTEX_CONVERSATION_MAX_TURNS": str(conversation_max_turns),
        }
        apply_env_updates(updates)
        if env_file is not None:
            upsert_env_file(Path(env_file), updates)
        return updates

    @property
    def vertex(self) -> VertexClient:
        """Lazily construct the underlying Vertex SDK wrapper."""

        if self._vertex is None:
            self._vertex = VertexClient(
                project=self.settings.google_cloud_project,
                location=self.settings.google_cloud_location,
            )
        return self._vertex

    def list_folders(self) -> list[DriveFolderOption]:
        """List Drive folders visible to the authenticated account."""

        service = build_drive_service(self.settings)
        return list_accessible_folders(service)

    def get_folders(self) -> list[dict[str, str | None]]:
        """Return visible Drive folders in a serialization-friendly structure."""

        return [
            {
                "folder_id": folder.folder_id,
                "name": folder.name,
                "web_view_link": folder.web_view_link,
            }
            for folder in self.list_folders()
        ]

    def choose_folder(
        self,
        *,
        persist: bool = True,
        env_file: Path | str | None = DEFAULT_ENV_FILE,
    ) -> DriveFolderOption:
        """Interactively choose a visible Drive folder and optionally persist it."""

        if not sys.stdin.isatty():
            raise ConfigurationError(
                "Interactive folder selection requires a terminal."
            )

        folders = self.list_folders()
        if not folders:
            raise RuntimeError(
                "No Google Drive folders were visible to the authenticated account."
            )

        selected = _choose_folder_option(
            folders,
            current_folder_id=self.settings.default_folder_id,
        )
        self.settings.default_folder_id = selected.folder_id
        apply_env_updates({"GOOGLE_DRIVE_FOLDER_ID": selected.folder_id})

        if persist:
            env_path = DEFAULT_ENV_FILE if env_file is None else Path(env_file)
            upsert_env_file(env_path, {"GOOGLE_DRIVE_FOLDER_ID": selected.folder_id})

        return selected

    def setup_auth(
        self,
        *,
        vertex_auth_mode: str = "adc",
        drive_auth_mode: str = "oauth",
    ) -> AuthSetupSummary:
        """Create or verify local auth state and enable required Google APIs."""

        return complete_auth_setup(
            settings=self.settings,
            vertex_auth_mode=vertex_auth_mode,
            drive_auth_mode=drive_auth_mode,
        )

    def auth(
        self,
        *,
        vertex_auth_mode: str = "adc",
        drive_auth_mode: str = "oauth",
    ) -> AuthSetupSummary:
        """Alias for `setup_auth()` with a shorter, more natural library name."""

        return self.setup_auth(
            vertex_auth_mode=vertex_auth_mode,
            drive_auth_mode=drive_auth_mode,
        )

    def load_index(self, *, index_dir: Path | str | None = None) -> LocalIndex:
        """Load the local vector index from disk."""

        return LocalIndex.load(self._resolve_index_dir(index_dir))

    def build_retriever(
        self,
        *,
        index_dir: Path | str | None = None,
    ) -> DriveCorpusRetriever:
        """Create a retriever bound to the current settings and local index."""

        return DriveCorpusRetriever(index=self.load_index(index_dir=index_dir), vertex=self.vertex)

    def sync(
        self,
        *,
        folder_id: str | None = None,
        index_dir: Path | str | None = None,
        recursive: bool = True,
        chunk_size_tokens: int = 350,
        chunk_overlap_tokens: int = 60,
        batch_size: int = 5,
        interactive: bool = False,
        env_file: Path | str | None = DEFAULT_ENV_FILE,
    ) -> SyncStats:
        """Rebuild the local vector index from a Drive folder."""

        return sync_folder(
            settings=self.settings,
            folder_id=self._resolve_folder_id(
                folder_id,
                interactive=interactive,
                env_file=env_file,
            ),
            index_dir=self._resolve_index_dir(index_dir),
            recursive=recursive,
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            batch_size=batch_size,
        )

    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        refresh: bool = False,
        folder_id: str | None = None,
        index_dir: Path | str | None = None,
        recursive: bool = True,
        chunk_size_tokens: int = 350,
        chunk_overlap_tokens: int = 60,
        batch_size: int = 5,
        model: str | None = None,
        temperature: float | None = None,
        conversation_max_turns: int | None = None,
        conversation_history: Sequence[tuple[str, str]] | None = None,
    ) -> RetrievalAnswer:
        """Answer one question against the indexed Drive corpus."""

        resolved_index_dir = self._resolve_index_dir(index_dir)
        if refresh:
            self.sync(
                folder_id=folder_id,
                index_dir=resolved_index_dir,
                recursive=recursive,
                chunk_size_tokens=chunk_size_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
                batch_size=batch_size,
            )

        retriever = self.build_retriever(index_dir=resolved_index_dir)
        return retriever.answer(
            question,
            model=model or self.settings.gemini_model,
            default_top_k=top_k or self.settings.default_top_k,
            temperature=(
                self.settings.gemini_temperature
                if temperature is None
                else temperature
            ),
            conversation_max_turns=(
                self.settings.conversation_max_turns
                if conversation_max_turns is None
                else conversation_max_turns
            ),
            conversation_history=conversation_history,
        )

    def open_chat(
        self,
        *,
        index_dir: Path | str | None = None,
        top_k: int | None = None,
        model: str | None = None,
        temperature: float | None = None,
        conversation_max_turns: int | None = None,
    ) -> DriveVertexChatSession:
        """Create a reusable chat session over the current local index."""

        return DriveVertexChatSession(
            client=self,
            index_dir=self._resolve_index_dir(index_dir) if index_dir is not None else None,
            top_k=top_k,
            model=model,
            temperature=temperature,
            conversation_max_turns=conversation_max_turns,
        )

    def status(self, *, index_dir: Path | str | None = None) -> DriveVertexStatus:
        """Return library-friendly status information for the local index."""

        resolved_index_dir = self._resolve_index_dir(index_dir)
        index = LocalIndex.load(resolved_index_dir)
        return DriveVertexStatus(
            index_dir=resolved_index_dir,
            manifest=index.manifest,
            gemini_model=self.settings.gemini_model,
            gemini_temperature=self.settings.gemini_temperature,
            configured_embedding_model=self.settings.embedding_model,
            configured_embedding_dimensions=self.settings.embedding_dimensions,
            default_top_k=self.settings.default_top_k,
            conversation_max_turns=self.settings.conversation_max_turns,
        )

    def _resolve_folder_id(
        self,
        folder_id: str | None,
        *,
        interactive: bool = False,
        env_file: Path | str | None = DEFAULT_ENV_FILE,
    ) -> str:
        """Resolve an explicit folder id or fall back to the configured default."""

        if folder_id:
            return folder_id
        if self.settings.default_folder_id:
            return self.settings.default_folder_id
        if interactive:
            return self.choose_folder(env_file=env_file).folder_id
        raise ConfigurationError(
            "No Google Drive folder is configured. Pass `folder_id=`, set "
            "GOOGLE_DRIVE_FOLDER_ID, or use `interactive=True`."
        )

    def _resolve_index_dir(self, index_dir: Path | str | None) -> Path:
        """Resolve an explicit index directory or fall back to the configured default."""

        if index_dir is None:
            return self.settings.index_dir
        return Path(index_dir)


def _stringify_env_value(value: Path | str | None) -> str:
    """Convert optional path-like values into the dotenv string form used by the library."""

    if value is None:
        return ""
    return str(value)


def _run_auth_cli(env_file: Path) -> None:
    """Run the interactive auth CLI flow using the current Python interpreter."""

    subprocess.run(
        [
            sys.executable,
            "-m",
            "drive_vertex_cli",
            "auth",
            "--env-file",
            str(env_file),
        ],
        check=True,
    )


def _choose_folder_option(
    folders: Sequence[DriveFolderOption],
    *,
    current_folder_id: str | None,
) -> DriveFolderOption:
    """Render a folder picker and return the selected option."""

    table = Table(
        title="Available Google Drive Folders",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    table.add_column("#", justify="right", style="bold cyan", no_wrap=True)
    table.add_column("Folder", style="bold")
    table.add_column("Default", style="green", no_wrap=True)
    table.add_column("Open")

    current_index: int | None = None
    for index, folder in enumerate(folders, start=1):
        default_marker = ""
        if folder.folder_id == current_folder_id:
            current_index = index
            default_marker = "current"
        table.add_row(
            str(index),
            folder.name,
            default_marker,
            folder.web_view_link or "",
        )

    CONSOLE.print(table)
    default_index = str(current_index or 1)

    while True:
        choice = input(f"Choose folder number [{default_index}]: ").strip() or default_index
        if choice.isdigit():
            selected_index = int(choice)
            if 1 <= selected_index <= len(folders):
                selected = folders[selected_index - 1]
                CONSOLE.print(
                    Panel(
                        f"Selected [bold]{selected.name}[/bold].",
                        title="Folder Selected",
                        border_style="green",
                        box=box.ROUNDED,
                        padding=(0, 1),
                    )
                )
                return selected

        ERROR_CONSOLE.print(
            Panel(
                "Enter a folder number from the list.",
                title="Error",
                border_style="red",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )
