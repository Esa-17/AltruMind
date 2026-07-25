import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import chunk_text, CHUNK_SIZE, CHUNK_OVERLAP  # noqa: E402


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