from __future__ import annotations

import re
from dataclasses import dataclass

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(slots=True)
class TextChunk:
    """A token-bounded chunk ready for embedding."""

    text: str
    token_count: int


def normalize_text(text: str) -> str:
    """Normalize line endings and trim surrounding whitespace."""

    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def count_tokens(text: str) -> int:
    """Approximate token count with a lightweight regex tokenizer."""

    return len(list(TOKEN_PATTERN.finditer(text)))


def chunk_text(
    text: str,
    *,
    max_tokens: int = 350,
    overlap_tokens: int = 60,
    min_tokens: int = 25,
) -> list[TextChunk]:
    """Split extracted document text into overlapping chunks for retrieval."""

    normalized = normalize_text(text)
    if not normalized:
        return []

    matches = list(TOKEN_PATTERN.finditer(normalized))
    if not matches:
        compact = WHITESPACE_PATTERN.sub(" ", normalized).strip()
        return [TextChunk(text=compact, token_count=1)] if compact else []

    chunks: list[TextChunk] = []
    step = max(max_tokens - overlap_tokens, 1)
    total_tokens = len(matches)
    start_index = 0

    while start_index < total_tokens:
        end_index = min(start_index + max_tokens, total_tokens)
        start_char = matches[start_index].start()
        end_char = matches[end_index - 1].end()
        window = normalized[start_char:end_char].strip()
        token_count = end_index - start_index

        if token_count >= min_tokens or end_index == total_tokens or not chunks:
            chunks.append(TextChunk(text=window, token_count=token_count))

        if end_index == total_tokens:
            break
        start_index += step

    if len(chunks) >= 2 and chunks[-1].token_count < min_tokens:
        merged_text = f"{chunks[-2].text}\n\n{chunks[-1].text}".strip()
        merged_token_count = count_tokens(merged_text)
        chunks[-2] = TextChunk(text=merged_text, token_count=merged_token_count)
        chunks.pop()

    return chunks
