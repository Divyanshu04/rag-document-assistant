
from pathlib import Path
import hashlib
from typing import Any

import chromadb


class VectorStore:
    """Persistent vector storage and similarity search using ChromaDB."""

    def __init__(
            self,
            persist_directory: str | Path = "data/chroma_db",
            collection_name: str = "documents",
    ):
        self.persist_directory = str(persist_directory)

        client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        self.collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _chunk_id(chunk: dict[str, Any]) -> str:
        """Generate a stable ID for a document chunk."""
        identity = (
            f"{chunk['source']}|"
            f"{chunk.get('page')}|"
            f"{chunk['chunk_index']}|"
            f"{chunk['text']}"
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def add_chunks(
            self,
            chunks: list[dict[str, Any]],
            embeddings: list[list[float]],
    ) -> int:
        """Store chunks, embeddings, and searchable metadata."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings."
            )

        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            ids.append(self._chunk_id(chunk))
            documents.append(chunk["text"])

            metadata = {
                "source": str(chunk["source"]),
                "chunk_index": int(chunk["chunk_index"]),
            }

            if chunk.get("page") is not None:
                metadata["page"] = int(chunk["page"])

            metadatas.append(metadata)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        return len(chunks)

    def search(
            self,
            query_embedding: list[float],
            top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Retrieve the most similar document chunks."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if not query_embedding:
            raise ValueError("Query embedding cannot be empty")

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        matches = []

        for i, document in enumerate(result["documents"][0]):
            matches.append(
                {
                    "text": document,
                    "metadata": result["metadatas"][0][i],
                    "distance": result["distances"][0][i],
                }
            )

        return matches

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self.collection.count()
