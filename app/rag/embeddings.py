import hashlib
import math
from typing import Any

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from app.config.settings import settings


class EmbeddingService:
    def __init__(self, model_name: str | None = None) -> None:
        self.backend = settings.embedding_backend
        self.model_name = model_name or settings.embedding_model_name
        self.dimension = settings.embedding_dimension
        self.model: SentenceTransformer | None = None
        self.client: OpenAI | None = None

        if self.backend == "sentence_transformers":
            self.model = SentenceTransformer(self.model_name)
        elif self.backend == "openai_compatible":
            if not settings.embedding_base_url:
                raise ValueError("EMBEDDING_BASE_URL is required when EMBEDDING_BACKEND=openai_compatible")
            if not settings.embedding_api_key:
                raise ValueError("EMBEDDING_API_KEY is required when EMBEDDING_BACKEND=openai_compatible")
            self.client = OpenAI(base_url=settings.embedding_base_url, api_key=settings.embedding_api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.backend == "hash":
            return [self._hash_embedding(text) for text in texts]
        if self.backend == "openai_compatible":
            return self._openai_compatible_embed(texts)
        assert self.model is not None
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        if self.backend == "hash":
            return self._hash_embedding(text)
        if self.backend == "openai_compatible":
            return self._openai_compatible_embed([text])[0]
        assert self.model is not None
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def _openai_compatible_embed(self, texts: list[str]) -> list[list[float]]:
        assert self.client is not None
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        sorted_data = sorted(response.data, key=lambda item: item.index)
        return [self._normalize_vector(item.embedding) for item in sorted_data]

    def _hash_embedding(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = self._tokenize(text)

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dimension
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vector[index] += sign

        return self._normalize_vector(vector)

    @staticmethod
    def _normalize_vector(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        normalized = text.lower().replace("\n", " ")
        tokens = [token.strip() for token in normalized.split() if token.strip()]
        if tokens:
            return tokens
        return [char for char in normalized if not char.isspace()]
