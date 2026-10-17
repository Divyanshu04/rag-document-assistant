
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    """Generate vector embeddings for text."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Convert multiple texts into embedding vectors."""
        if not texts:
            return []

        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Convert a single query into an embedding vector."""
        if not query.strip():
            raise ValueError("Query cannot be empty")

        return self.embed_texts([query])[0]
