from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from app.config.settings import settings
from app.rag.schema import ChunkDocument, RetrievalResult


MetadataFilters = dict[str, str | int | float | bool]


class ChromaVectorStore:
    def __init__(self, collection_name: str = "agentic_rag") -> None:
        self.client = chromadb.PersistentClient(path=str(settings.vector_db_path))
        self.collection_name = collection_name
        self.collection: Collection = self.client.get_or_create_collection(name=collection_name)

    def reset_collection(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_chunks(self, chunks: list[ChunkDocument], embeddings: list[list[float]]) -> None:
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            metadatas=[self._metadata_for_chunk(chunk) for chunk in chunks],
            embeddings=embeddings,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: MetadataFilters | None = None,
    ) -> list[RetrievalResult]:
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters or None,
        )
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        retrievals: list[RetrievalResult] = []
        for chunk_id, content, metadata, distance in zip(ids, docs, metas, distances):
            metadata = metadata or {}
            retrievals.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    content=content,
                    score=float(distance) if distance is not None else 0.0,
                    source_file=str(metadata.get("source_file", "unknown")),
                    metadata=metadata,
                )
            )
        return retrievals

    @staticmethod
    def _metadata_for_chunk(chunk: ChunkDocument) -> dict[str, Any]:
        metadata = {
            key: value
            for key, value in chunk.metadata.items()
            if isinstance(value, (str, int, float, bool))
        }
        metadata["source_file"] = chunk.source_file
        metadata["file_type"] = chunk.file_type
        return metadata
