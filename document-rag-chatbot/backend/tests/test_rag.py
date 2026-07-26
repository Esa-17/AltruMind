import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import rag  # noqa: E402
from rag import chunk_text, CHUNK_SIZE, CHUNK_OVERLAP, VectorStore, EMBED_DIM  # noqa: E402


def test_chunk_text_empty_page_produces_no_chunks():
    assert chunk_text([("", 1)]) == []
    assert chunk_text([("   ", 1)]) == []


def test_chunk_text_short_text_single_chunk():
    text = "A short paragraph that fits in one chunk."
    chunks = chunk_text([(text, 1)])
    assert len(chunks) == 1
    assert chunks[0]["text"] == text
    assert chunks[0]["page"] == 1


def test_chunk_text_long_text_splits_with_overlap():
    text = "x" * (CHUNK_SIZE * 3)
    chunks = chunk_text([(text, 5)])

    assert len(chunks) > 1
    for c in chunks:
        assert len(c["text"]) <= CHUNK_SIZE
        assert c["page"] == 5

    # Overlap check: the tail of chunk N should reappear at the start
    # of chunk N+1, sized by CHUNK_OVERLAP.
    first_tail = chunks[0]["text"][-CHUNK_OVERLAP:]
    second_head = chunks[1]["text"][:CHUNK_OVERLAP]
    assert first_tail == second_head


def test_chunk_text_preserves_multiple_pages_independently():
    pages = [("First page content here.", 1), ("Second page content here.", 2)]
    chunks = chunk_text(pages)
    pages_seen = {c["page"] for c in chunks}
    assert pages_seen == {1, 2}


# ---------------------------------------------------------------------
# VectorStore — embeds via the Gemini API (mocked here, no real network
# call). This is the path that used to call a local sentence-transformers
# model; these tests confirm the swap didn't break add/search behavior.
# ---------------------------------------------------------------------

def test_vectorstore_add_and_search_roundtrip():
    store = VectorStore()
    chunks = [
        {"text": "Freelancers lose money on late invoices.", "page": 1, "source": "Page 1"},
        {"text": "Our pricing is nine dollars a month.", "page": 2, "source": "Page 2"},
    ]

    with patch.object(rag, "_get_client") as mock_get_client:
        mock_client = mock_get_client.return_value
        mock_client.models.embed_content.return_value = type(
            "R", (), {"embeddings": [
                type("E", (), {"values": [0.1] * EMBED_DIM})(),
                type("E", (), {"values": [0.2] * EMBED_DIM})(),
            ]}
        )()
        store.add_chunks(chunks)

    assert store.index.ntotal == 2
    assert not store.is_empty()

    with patch.object(rag, "_get_client") as mock_get_client:
        mock_client = mock_get_client.return_value
        mock_client.models.embed_content.return_value = type(
            "R", (), {"embeddings": [type("E", (), {"values": [0.1] * EMBED_DIM})()]}
        )()
        results = store.search("late invoices", top_k=2)

    assert len(results) <= 2
    for r in results:
        assert "score" in r


def test_embed_texts_returns_empty_array_for_empty_input():
    result = rag._embed_texts([])
    assert result.shape == (0, EMBED_DIM)