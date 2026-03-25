from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class ChunkRecord:
    """A single embedded document chunk stored in the local index."""

    chunk_id: str
    file_id: str
    file_name: str
    drive_path: str
    mime_type: str
    modified_time: str | None
    web_view_link: str | None
    chunk_index: int
    token_count: int
    text: str


@dataclass(slots=True)
class IndexManifest:
    """Metadata describing how a local index was built."""

    version: int
    synced_at: str
    folder_id: str
    embedding_model: str
    embedding_dimensions: int | None
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    file_count: int
    chunk_count: int


@dataclass(slots=True)
class SearchHit:
    """A retrieved chunk plus its similarity score."""

    score: float
    record: ChunkRecord


class LocalIndex:
    """On-disk NumPy-backed vector index used by the library and CLI."""

    def __init__(
        self,
        manifest: IndexManifest,
        chunks: list[ChunkRecord],
        embeddings: np.ndarray,
    ) -> None:
        self.manifest = manifest
        self.chunks = chunks
        self.embeddings = embeddings.astype(np.float32)

    @classmethod
    def load(cls: type["LocalIndex"], index_dir: Path) -> "LocalIndex":
        """Load a previously saved local index from disk."""

        manifest_path = index_dir / "manifest.json"
        chunks_path = index_dir / "chunks.jsonl"
        embeddings_path = index_dir / "embeddings.npy"

        if not manifest_path.exists() or not chunks_path.exists() or not embeddings_path.exists():
            raise FileNotFoundError(
                f"Index files not found in {index_dir}. Run `drive-vertex sync` first."
            )

        manifest = IndexManifest(**json.loads(manifest_path.read_text()))
        chunks = [
            ChunkRecord(**json.loads(line))
            for line in chunks_path.read_text().splitlines()
            if line.strip()
        ]
        embeddings = np.load(embeddings_path)
        return cls(manifest, chunks, embeddings)

    @classmethod
    def build(
        cls: type["LocalIndex"],
        *,
        folder_id: str,
        embedding_model: str,
        embedding_dimensions: int | None,
        chunk_size_tokens: int,
        chunk_overlap_tokens: int,
        chunks: list[ChunkRecord],
        embeddings: np.ndarray,
        file_count: int,
    ) -> "LocalIndex":
        """Construct an in-memory index from chunk records and raw embeddings."""

        manifest = IndexManifest(
            version=1,
            synced_at=datetime.now(timezone.utc).isoformat(),
            folder_id=folder_id,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            file_count=file_count,
            chunk_count=len(chunks),
        )
        return cls(manifest, chunks, normalize_embeddings(embeddings))

    def save(self, index_dir: Path) -> None:
        """Persist the manifest, chunks, and normalized embeddings to disk."""

        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / "manifest.json").write_text(json.dumps(asdict(self.manifest), indent=2))
        (index_dir / "chunks.jsonl").write_text(
            "\n".join(json.dumps(asdict(chunk), ensure_ascii=True) for chunk in self.chunks)
        )
        np.save(index_dir / "embeddings.npy", self.embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[SearchHit]:
        """Return the highest-scoring chunks for a query embedding."""

        if self.embeddings.size == 0:
            return []

        normalized_query = normalize_embeddings(query_embedding.reshape(1, -1))[0]
        scores = self.embeddings @ normalized_query
        limit = min(top_k, len(self.chunks))
        indices = np.argsort(scores)[::-1][:limit]
        return [
            SearchHit(score=float(scores[index]), record=self.chunks[index])
            for index in indices
        ]


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings so dot products behave like cosine similarity."""

    values = embeddings.astype(np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)

    # Zero vectors are preserved by treating their norm as 1.0.
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return values / norms
