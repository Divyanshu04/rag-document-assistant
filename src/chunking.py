
from typing import Any


def chunk_text(
        text: str,
        chunk_size: int = 500,
        overlap: int = 50,
) -> list[str]:
    """Split text into overlapping character-based chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def chunk_records(
        records: list[dict[str, Any]],
        chunk_size: int = 500,
        overlap: int = 50,
) -> list[dict[str, Any]]:
    """Chunk ingested records while preserving their metadata."""

    chunked_records = []

    for record in records:
        chunks = chunk_text(
            record["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for index, chunk in enumerate(chunks):
            chunked_records.append(
                {
                    "source": record["source"],
                    "page": record["page"],
                    "chunk_index": index,
                    "text": chunk,
                }
            )

    return chunked_records
