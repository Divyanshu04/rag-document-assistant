
import pytest

from src.chunking import chunk_text, chunk_records


def test_chunk_text_splits_long_text():
    text = "A" * 1200

    chunks = chunk_text(text, chunk_size=500, overlap=50)

    assert len(chunks) == 3
    assert [len(chunk) for chunk in chunks] == [500, 500, 300]


def test_chunk_text_preserves_overlap():
    text = "ABCDEFGHIJKLMNOP"

    chunks = chunk_text(text, chunk_size=8, overlap=2)

    assert chunks == [
        "ABCDEFGH",
        "GHIJKLMN",
        "MNOP",
    ]


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_invalid_chunk_size_raises_error():
    with pytest.raises(ValueError):
        chunk_text("Some text", chunk_size=0)


def test_invalid_overlap_raises_error():
    with pytest.raises(ValueError):
        chunk_text("Some text", chunk_size=10, overlap=10)


def test_chunk_records_preserves_metadata():
    records = [
        {
            "source": "policy.pdf",
            "page": 3,
            "text": "A" * 600,
        }
    ]

    chunks = chunk_records(
        records,
        chunk_size=500,
        overlap=50,
    )

    assert len(chunks) == 2
    assert chunks[0]["source"] == "policy.pdf"
    assert chunks[0]["page"] == 3
    assert chunks[0]["chunk_index"] == 0
    assert chunks[1]["chunk_index"] == 1
    assert len(chunks[1]["text"]) == 150
