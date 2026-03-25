from drive_vertex_cli.chunking import chunk_text, count_tokens


def test_chunk_text_creates_overlapping_windows():
    text = " ".join(f"token{i}" for i in range(120))
    chunks = chunk_text(text, max_tokens=50, overlap_tokens=10, min_tokens=5)

    assert len(chunks) == 3
    assert chunks[0].token_count == 50
    assert chunks[1].token_count == 50
    assert chunks[2].token_count == 40


def test_count_tokens_counts_words_and_punctuation():
    assert count_tokens("Hello, world!") == 4
