
import json
from pathlib import Path

from src.embeddings import EmbeddingModel


def main():
    project_root = Path(__file__).resolve().parent
    chunks_file = project_root / "data" / "processed" / "chunks.jsonl"

    if not chunks_file.is_file():
        raise FileNotFoundError(
            f"Chunk file not found: {chunks_file}. "
            "Run python -m src.process_documents first."
        )

    chunks = []

    with chunks_file.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    if not chunks:
        print("No chunks found.")
        return

    model = EmbeddingModel()

    texts = [chunk["text"] for chunk in chunks]
    vectors = model.embed_texts(texts)

    print(f"Chunks embedded: {len(vectors)}")
    print(f"Vector dimensions: {len(vectors[0])}")
    print(f"Source of first chunk: {chunks[0]['source']}")
    print(f"First 10 vector values: {vectors[0][:10]}")


if __name__ == "__main__":
    main()
