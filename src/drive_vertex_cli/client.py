from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from drive_vertex_cli.config import ConfigurationError, Settings, load_settings
from drive_vertex_cli.drive_client import DriveFolderOption, build_drive_service, list_accessible_folders
from drive_vertex_cli.index_store import IndexManifest, LocalIndex
from drive_vertex_cli.indexer import SyncStats, sync_folder
from drive_vertex_cli.retrieval import DriveCorpusRetriever, RetrievalAnswer
from drive_vertex_cli.vertex_client import VertexClient


@dataclass(slots=True)
class DriveVertexStatus:
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
    client: "DriveVertexClient"
    index_dir: Path | None = None
    top_k: int | None = None
    model: str | None = None
    temperature: float | None = None
    conversation_max_turns: int | None = None
    history: list[tuple[str, str]] = field(default_factory=list)

    def ask(self, question: str, *, top_k: int | None = None) -> RetrievalAnswer:
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
        self.history.clear()


class DriveVertexClient:
    def __init__(
        self,
        settings: Settings,
        *,
        vertex: VertexClient | None = None,
    ) -> None:
        self.settings = settings
        self._vertex = vertex

    @classmethod
    def from_env(cls, *, require_project: bool = True) -> "DriveVertexClient":
        return cls(load_settings(require_project=require_project))

    @property
    def vertex(self) -> VertexClient:
        if self._vertex is None:
            self._vertex = VertexClient(
                project=self.settings.google_cloud_project,
                location=self.settings.google_cloud_location,
            )
        return self._vertex

    def list_folders(self) -> list[DriveFolderOption]:
        service = build_drive_service(self.settings)
        return list_accessible_folders(service)

    def load_index(self, *, index_dir: Path | str | None = None) -> LocalIndex:
        return LocalIndex.load(self._resolve_index_dir(index_dir))

    def build_retriever(
        self,
        *,
        index_dir: Path | str | None = None,
    ) -> DriveCorpusRetriever:
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
    ) -> SyncStats:
        return sync_folder(
            settings=self.settings,
            folder_id=self._resolve_folder_id(folder_id),
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
        return DriveVertexChatSession(
            client=self,
            index_dir=self._resolve_index_dir(index_dir) if index_dir is not None else None,
            top_k=top_k,
            model=model,
            temperature=temperature,
            conversation_max_turns=conversation_max_turns,
        )

    def status(self, *, index_dir: Path | str | None = None) -> DriveVertexStatus:
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

    def _resolve_folder_id(self, folder_id: str | None) -> str:
        if folder_id:
            return folder_id
        if self.settings.default_folder_id:
            return self.settings.default_folder_id
        raise ConfigurationError(
            "No Google Drive folder is configured. Pass `folder_id=` or set GOOGLE_DRIVE_FOLDER_ID."
        )

    def _resolve_index_dir(self, index_dir: Path | str | None) -> Path:
        if index_dir is None:
            return self.settings.index_dir
        return Path(index_dir)
