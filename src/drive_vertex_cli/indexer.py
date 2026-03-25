from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from googleapiclient.discovery import Resource

from drive_vertex_cli.chunking import chunk_text
from drive_vertex_cli.config import Settings
from drive_vertex_cli.drive_client import (
    DriveDocument,
    build_drive_service,
    download_document,
    list_documents,
)
from drive_vertex_cli.extractors import UnsupportedFileTypeError, extract_text
from drive_vertex_cli.index_store import ChunkRecord, LocalIndex
from drive_vertex_cli.vertex_client import VertexClient


@dataclass(slots=True)
class SyncStats:
    """Summary of a completed sync run."""

    indexed_file_count: int
    skipped_file_count: int
    chunk_count: int
    skipped_reasons: list[str]


def sync_folder(
    *,
    settings: Settings,
    folder_id: str,
    index_dir: Path,
    recursive: bool,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    batch_size: int,
) -> SyncStats:
    """Read a Drive folder, chunk its text, and write a fresh local vector index."""

    service = build_drive_service(settings)
    vertex = VertexClient(
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
    )

    documents = list_documents(service, folder_id, recursive=recursive)
    chunk_records: list[ChunkRecord] = []
    texts_for_embedding: list[str] = []
    skipped_reasons: list[str] = []
    indexed_files = 0

    if not documents:
        raise RuntimeError(
            "The selected Google Drive folder is reachable, but no child files were visible "
            "to the authenticated account. Confirm the folder is not empty, that you "
            "authorized the correct Google account, and that the files are inside this folder."
        )

    for document in documents:
        try:
            chunk_records_for_document, embedding_texts = _document_to_chunks(
                service=service,
                document=document,
                chunk_size_tokens=chunk_size_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
            )
        except UnsupportedFileTypeError as exc:
            skipped_reasons.append(str(exc))
            continue
        except Exception as exc:
            skipped_reasons.append(f"Failed to process {document.drive_path}: {exc}")
            continue

        if not chunk_records_for_document:
            skipped_reasons.append(f"No usable text found in {document.drive_path}")
            continue

        indexed_files += 1
        chunk_records.extend(chunk_records_for_document)
        texts_for_embedding.extend(embedding_texts)

    if not chunk_records:
        details = "\n".join(f"- {reason}" for reason in skipped_reasons[:10])
        suffix = ""
        if len(skipped_reasons) > 10:
            suffix = f"\n... and {len(skipped_reasons) - 10} more."
        raise RuntimeError(
            "No supported text content was found in the selected Drive folder.\n"
            f"{details}{suffix}"
        )

    embeddings = vertex.embed_texts(
        texts_for_embedding,
        model=settings.embedding_model,
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=settings.embedding_dimensions,
        batch_size=batch_size,
    )

    index = LocalIndex.build(
        folder_id=folder_id,
        embedding_model=settings.embedding_model,
        embedding_dimensions=settings.embedding_dimensions,
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
        chunks=chunk_records,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        file_count=indexed_files,
    )
    index.save(index_dir)

    return SyncStats(
        indexed_file_count=indexed_files,
        skipped_file_count=len(documents) - indexed_files,
        chunk_count=len(chunk_records),
        skipped_reasons=skipped_reasons,
    )


def _document_to_chunks(
    *,
    service: Resource,
    document: DriveDocument,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> tuple[list[ChunkRecord], list[str]]:
    """Convert one Drive document into chunk records and embedding payloads."""

    name, mime_type, payload = download_document(service, document)
    text = extract_text(name, mime_type, payload)
    chunks = chunk_text(
        text,
        max_tokens=chunk_size_tokens,
        overlap_tokens=chunk_overlap_tokens,
    )

    records: list[ChunkRecord] = []
    embedding_texts: list[str] = []

    for chunk_index, chunk in enumerate(chunks):
        chunk_id = f"{document.file_id}:{chunk_index}"
        records.append(
            ChunkRecord(
                chunk_id=chunk_id,
                file_id=document.file_id,
                file_name=document.name,
                drive_path=document.drive_path,
                mime_type=document.mime_type,
                modified_time=document.modified_time,
                web_view_link=document.web_view_link,
                chunk_index=chunk_index,
                token_count=chunk.token_count,
                text=chunk.text,
            )
        )
        embedding_texts.append(chunk.text)

    return records, embedding_texts
