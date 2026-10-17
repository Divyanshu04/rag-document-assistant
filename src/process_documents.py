
import json
from pathlib import Path

from src.ingest import load_document
from src.chunking import chunk_records


def main():
    project_root = Path(__file__).resolve().parent.parent
    documents_dir = project_root / "data" / "documents"
    output_dir = project_root / "data" / "processed"
    output_file = output_dir / "chunks.jsonl"

    supported_extensions = {".txt", ".pdf"}

    if not documents_dir.is_dir():
        raise FileNotFoundError(
            f"Documents directory not found: {documents_dir}"
        )

    # 1. Read all real TXT and PDF documents.
    records = []

    for file_path in sorted(documents_dir.iterdir()):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in supported_extensions:
            continue

        extracted = load_document(file_path)
        records.extend(extracted)

        print(
            f"Ingested: {file_path.name} "
            f"({len(extracted)} records)"
        )

    if not records:
        print("No text records found. Add a TXT or PDF to data/documents/.")
        return

    # 2. Chunk the extracted records.
    chunks = chunk_records(
        records,
        chunk_size=500,
        overlap=50,
    )

    # 3. Save chunks as JSON Lines.
    output_dir.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    print("\nProcessing complete!")
    print(f"Extracted records: {len(records)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Saved to: {output_file}")

    # 4. Preview the first real chunk.
    if chunks:
        first_chunk = chunks[0]
        print("\nFirst chunk preview:")
        print(f"Source: {first_chunk['source']}")
        print(f"Page: {first_chunk['page']}")
        print(f"Chunk index: {first_chunk['chunk_index']}")
        print(first_chunk["text"][:300])


if __name__ == "__main__":
    main()
