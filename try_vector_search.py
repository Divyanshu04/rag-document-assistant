
import json
from pathlib import Path

from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
#from src.generator import generate_answer
from src.generate import generate_answer

def main():
    project_root = Path(__file__).resolve().parent

    chunks_file = (
            project_root / "data" / "processed" / "chunks.jsonl"
    )

    if not chunks_file.is_file():
        raise FileNotFoundError(
            "Chunks not found. Run "
            "'python -m src.process_documents' first."
        )

    chunks = []

    with chunks_file.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    if not chunks:
        print("No chunks found.")
        return

    # Generate embeddings for the actual document chunks.
    model = EmbeddingModel()
    vectors = model.embed_texts(
        [chunk["text"] for chunk in chunks]
    )

    # Store vectors persistently.
    store = VectorStore(
        persist_directory=project_root / "data" / "chroma_db"
    )

    stored = store.add_chunks(chunks, vectors)

    print(f"Chunks processed: {len(chunks)}")
    print(f"Chunks upserted: {stored}")
    print(f"Total chunks in database: {store.count()}")

    # Ask a question about the documents.
    query = input(
        "\nAsk a question about your documents: "
    ).strip()

    if not query:
        print("No question entered.")
        return

    # Step 1: Retrieve relevant passages.
    query_vector = model.embed_query(query)
    results = store.search(query_vector, top_k=3)

    print("\nMost relevant passages:\n")

    if not results:
        print("No matching passages found.")
        return

    for rank, match in enumerate(results, start=1):
        metadata = match["metadata"]

        print(f"Result {rank}")
        print(f"Source: {metadata['source']}")
        print(f"Page: {metadata.get('page', 'N/A')}")
        print(f"Distance: {match['distance']:.4f}")
        print(f"Text: {match['text']}")
        print("-" * 60)

    # Step 2: Generate a grounded answer using Gemini.
    print("\nGenerating answer with Gemini...\n")

    answer = generate_answer(query, results)

    print("=" * 60)
    print("GEMINI ANSWER")
    print("=" * 60)
    print(answer)
    print("=" * 60)


if __name__ == "__main__":
    main()
