from __future__ import annotations

from itertools import islice
from typing import Any, Iterable

import numpy as np
from google import genai
from google.genai import types


class VertexClient:
    def __init__(self, *, project: str, location: str) -> None:
        self.client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
        )

    def embed_query(
        self,
        query: str,
        *,
        model: str,
        output_dimensionality: int | None,
    ) -> np.ndarray:
        return np.asarray(
            self.embed_texts(
                [query],
                model=model,
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=output_dimensionality,
                batch_size=1,
            )[0],
            dtype=np.float32,
        )

    def embed_texts(
        self,
        texts: list[str],
        *,
        model: str,
        task_type: str,
        output_dimensionality: int | None,
        batch_size: int,
    ) -> list[list[float]]:
        vectors: list[list[float]] = []
        effective_batch_size = (
            1 if model == "gemini-embedding-001" else min(max(batch_size, 1), 5)
        )
        config = types.EmbedContentConfig(task_type=task_type)
        if output_dimensionality:
            config.output_dimensionality = output_dimensionality

        for batch in batched(texts, effective_batch_size):
            response = self.client.models.embed_content(
                model=model,
                contents=batch,
                config=config,
            )
            vectors.extend(_coerce_embeddings(response))
        return vectors

    def generate_content(
        self,
        *,
        contents: str | list[types.Content],
        model: str,
        config: types.GenerateContentConfig,
    ):
        return self.client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

    @staticmethod
    def extract_text(response: Any) -> str:
        if (
            not getattr(response, "candidates", None)
            or not response.candidates[0].content
            or not response.candidates[0].content.parts
        ):
            return ""

        texts: list[str] = []
        for part in response.candidates[0].content.parts:
            if isinstance(part.text, str):
                if isinstance(part.thought, bool) and part.thought:
                    continue
                texts.append(part.text)
        return "".join(texts).strip()


def batched(values: Iterable[str], batch_size: int) -> Iterable[list[str]]:
    iterator = iter(values)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def _coerce_embeddings(response) -> list[list[float]]:
    embeddings = getattr(response, "embeddings", None)
    if embeddings is None:
        raise RuntimeError("Vertex AI did not return embeddings.")

    coerced: list[list[float]] = []
    for item in embeddings:
        values = getattr(item, "values", None)
        if values is None:
            values = item["values"]
        coerced.append(list(values))
    return coerced
