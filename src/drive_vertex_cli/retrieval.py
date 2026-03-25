from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from google.genai import types

from drive_vertex_cli.index_store import LocalIndex, SearchHit
from drive_vertex_cli.vertex_client import VertexClient

SYSTEM_PROMPT = """You answer questions only with evidence returned by the search_drive_corpus tool.
Call the tool before answering a new user question.
If the retrieved snippets are insufficient, say that the answer is not grounded in the indexed Google Drive corpus.
When you answer, cite source file names inline."""


@dataclass(slots=True)
class RetrievalAnswer:
    answer: str
    hits: list[SearchHit]


class DriveCorpusRetriever:
    def __init__(self, index: LocalIndex, vertex: VertexClient) -> None:
        self.index = index
        self.vertex = vertex

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        embedding = self.vertex.embed_query(
            query,
            model=self.index.manifest.embedding_model,
            output_dimensionality=self.index.manifest.embedding_dimensions,
        )
        return self.index.search(embedding, top_k=top_k)

    def answer(
        self,
        question: str,
        *,
        model: str,
        default_top_k: int = 5,
        temperature: float = 0.2,
        conversation_max_turns: int = 6,
        conversation_history: Sequence[tuple[str, str]] | None = None,
    ) -> RetrievalAnswer:
        latest_hits: list[SearchHit] = []

        def search_drive_corpus(query: str, top_k: int = default_top_k) -> dict:
            """Retrieve the most relevant snippets from the indexed Google Drive corpus.

            Args:
                query: The search string to run against the local Google Drive vector index.
                top_k: The maximum number of snippets to return.
            """

            nonlocal latest_hits
            latest_hits = self.search(query, top_k=top_k)
            return {
                "matches": [
                    {
                        "score": round(hit.score, 4),
                        "file_name": hit.record.file_name,
                        "drive_path": hit.record.drive_path,
                        "chunk_index": hit.record.chunk_index,
                        "web_view_link": hit.record.web_view_link,
                        "text": hit.record.text,
                    }
                    for hit in latest_hits
                ]
            }

        prompt = build_prompt(
            question,
            conversation_history or [],
            max_turns=conversation_max_turns,
        )
        contents: list[types.Content] = [
            types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
        ]
        initial_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=temperature,
            tools=[search_drive_corpus],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=["search_drive_corpus"],
                )
            ),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        response = self.vertex.generate_content(
            contents=contents,
            model=model,
            config=initial_config,
        )
        function_calls = response.function_calls or []
        if not function_calls:
            return RetrievalAnswer(answer=self.vertex.extract_text(response), hits=latest_hits)

        if response.candidates and response.candidates[0].content:
            contents.append(response.candidates[0].content)

        function_response_parts: list[types.Part] = []
        for function_call in function_calls:
            if function_call.name != "search_drive_corpus":
                raise RuntimeError(
                    f"Unexpected function call requested by model: {function_call.name}"
                )
            tool_response = search_drive_corpus(**(function_call.args or {}))
            function_response_parts.append(
                types.Part.from_function_response(
                    name=function_call.name,
                    response=tool_response,
                )
            )

        contents.append(types.Content(role="user", parts=function_response_parts))

        final_response = self.vertex.generate_content(
            contents=contents,
            model=model,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=temperature,
                tools=[search_drive_corpus],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="NONE")
                ),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        final_text = self.vertex.extract_text(final_response)
        if not final_text:
            raise RuntimeError("The model returned no final text after retrieval.")
        return RetrievalAnswer(answer=final_text, hits=latest_hits)


def build_prompt(
    question: str,
    conversation_history: Sequence[tuple[str, str]],
    *,
    max_turns: int = 6,
) -> str:
    if max_turns <= 0:
        return question

    turns = list(conversation_history)[-max_turns:]
    if not turns:
        return question

    lines = [
        "Conversation so far:",
    ]
    for user_message, assistant_message in turns:
        lines.append(f"User: {user_message}")
        lines.append(f"Assistant: {assistant_message}")

    lines.extend(
        [
            "",
            "Use the conversation only for context. Ground the answer in retrieved Drive snippets.",
            f"Current user question: {question}",
        ]
    )
    return "\n".join(lines)
