from types import SimpleNamespace

from google.genai import types

from drive_vertex_cli.index_store import ChunkRecord, SearchHit
from drive_vertex_cli.retrieval import DriveCorpusRetriever
from drive_vertex_cli.retrieval import build_prompt


def test_build_prompt_includes_recent_history_and_current_question():
    prompt = build_prompt(
        "What is the deadline?",
        [
            ("What is SQG?", "It is a project."),
            ("Who owns it?", "The platform team."),
        ],
    )

    assert "Conversation so far:" in prompt
    assert "User: What is SQG?" in prompt
    assert "Assistant: The platform team." in prompt
    assert "Current user question: What is the deadline?" in prompt


def test_build_prompt_returns_plain_question_without_history():
    assert build_prompt("Hello", []) == "Hello"


def test_build_prompt_ignores_history_when_max_turns_is_zero():
    prompt = build_prompt(
        "What is the deadline?",
        [("Earlier question", "Earlier answer")],
        max_turns=0,
    )

    assert prompt == "What is the deadline?"


def test_answer_executes_function_call_and_returns_final_text():
    function_call = types.FunctionCall(
        name="search_drive_corpus",
        args={"query": "SQG", "top_k": 1},
    )

    first_response = SimpleNamespace(
        function_calls=[function_call],
        candidates=[
            SimpleNamespace(
                content=types.Content(
                    role="model",
                    parts=[types.Part(function_call=function_call)],
                )
            )
        ],
    )
    second_response = SimpleNamespace(
        function_calls=[],
        candidates=[
            SimpleNamespace(
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="SQG is the indexed project.")],
                )
            )
        ],
    )

    class FakeVertex:
        def __init__(self):
            self.responses = [first_response, second_response]
            self.temperatures = []

        def generate_content(self, *, contents, model, config):
            self.temperatures.append(config.temperature)
            return self.responses.pop(0)

        @staticmethod
        def extract_text(response):
            return "".join(
                part.text
                for part in response.candidates[0].content.parts
                if getattr(part, "text", None)
            )

    vertex = FakeVertex()
    retriever = DriveCorpusRetriever(index=None, vertex=vertex)
    hit = SearchHit(
        score=0.99,
        record=ChunkRecord(
            chunk_id="1:0",
            file_id="1",
            file_name="sqg.txt",
            drive_path="sqg.txt",
            mime_type="text/plain",
            modified_time=None,
            web_view_link=None,
            chunk_index=0,
            token_count=3,
            text="SQG is the indexed project.",
        ),
    )
    retriever.search = lambda query, top_k=5: [hit]

    result = retriever.answer(
        "What does SQG do?",
        model="gemini-test",
        default_top_k=1,
        temperature=0.7,
        conversation_max_turns=4,
    )

    assert result.answer == "SQG is the indexed project."
    assert result.hits == [hit]
    assert vertex.temperatures == [0.7, 0.7]
