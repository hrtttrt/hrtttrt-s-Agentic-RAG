from app.config.settings import settings
from app.rag.embeddings import EmbeddingService
from app.rag.reranker import SimpleReranker
from app.rag.schema import RetrievalResult
from app.rag.vector_store import ChromaVectorStore, MetadataFilters


class Retriever:
    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: ChromaVectorStore | None = None,
        reranker: SimpleReranker | None = None,
    ) -> None:
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or ChromaVectorStore()
        self.reranker = reranker or SimpleReranker()

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: MetadataFilters | None = None,
    ) -> list[RetrievalResult]:
        final_top_k = top_k or settings.top_k
        candidate_k = min(max(final_top_k * 4, final_top_k), 20)
        query_embedding = self.embedding_service.embed_query(query)
        candidates = self.vector_store.search(query_embedding, candidate_k, filters=filters)
        reranked = self.reranker.rerank(query, candidates)
        return reranked[:final_top_k]
