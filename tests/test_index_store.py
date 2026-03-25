import numpy as np

from drive_vertex_cli.index_store import ChunkRecord, LocalIndex


def test_local_index_search_returns_best_match():
    chunks = [
        ChunkRecord(
            chunk_id="1:0",
            file_id="1",
            file_name="alpha.txt",
            drive_path="alpha.txt",
            mime_type="text/plain",
            modified_time=None,
            web_view_link=None,
            chunk_index=0,
            token_count=3,
            text="alpha topic",
        ),
        ChunkRecord(
            chunk_id="2:0",
            file_id="2",
            file_name="beta.txt",
            drive_path="beta.txt",
            mime_type="text/plain",
            modified_time=None,
            web_view_link=None,
            chunk_index=0,
            token_count=3,
            text="beta topic",
        ),
    ]

    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = LocalIndex.build(
        folder_id="folder",
        embedding_model="text-embedding-005",
        embedding_dimensions=2,
        chunk_size_tokens=350,
        chunk_overlap_tokens=60,
        chunks=chunks,
        embeddings=embeddings,
        file_count=2,
    )

    hits = index.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=1)
    assert hits[0].record.file_name == "alpha.txt"
