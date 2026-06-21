from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from app.loaders.router import LoaderRouter
from app.rag.embeddings import EmbeddingService
from app.rag.retriever import Retriever
from app.rag.schema import ChunkDocument, RawDocument, RetrievalResult
from app.rag.splitter import DocumentSplitter
from app.rag.vector_store import ChromaVectorStore


MetadataFilters = dict[str, str | int | float | bool]


class RAGTools:
    def __init__(self) -> None:
        self.loader_router = LoaderRouter()
        self.splitter = DocumentSplitter()
        self.embedding_service = EmbeddingService()
        self.vector_store = ChromaVectorStore()
        self.retriever = Retriever(
            embedding_service=self.embedding_service,
            vector_store=self.vector_store,
        )

    def load_documents(self, paths: list[str | Path]) -> list[RawDocument]:
        documents: list[RawDocument] = []
        for path in paths:
            file_path = Path(path)
            if file_path.is_dir():
                for child in file_path.rglob("*"):
                    if child.is_file():
                        documents.extend(self.loader_router.load(child))
            else:
                documents.extend(self.loader_router.load(file_path))
        return documents

    def split_documents(self, documents: list[RawDocument]) -> list[ChunkDocument]:
        return self.splitter.split(documents)

    def index_documents(self, chunks: list[ChunkDocument]) -> int:
        if not chunks:
            return 0
        embeddings = self.embedding_service.embed_documents([chunk.content for chunk in chunks])
        self.vector_store.add_chunks(chunks, embeddings)
        return len(chunks)

    def build_knowledge_base(self, paths: list[str | Path]) -> int:
        documents = self.load_documents(paths)
        chunks = self.split_documents(documents)
        return self.index_documents(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: MetadataFilters | None = None,
    ) -> list[RetrievalResult]:
        return self.retriever.search(query=query, top_k=top_k, filters=filters)

    @staticmethod
    def format_snippets(results: list[RetrievalResult]) -> str:
        lines: list[str] = []
        for idx, item in enumerate(results, start=1):
            metadata_text = ", ".join(f"{key}={value}" for key, value in item.metadata.items() if key not in {"source_file"})
            location = f" ({metadata_text})" if metadata_text else ""
            lines.append(f"[{idx}] 来源: {item.source_file}{location}\n{item.content}")
        return "\n\n".join(lines)


_default_rag_tools = RAGTools()


@tool("rag_retrieve")
def rag_retrieve_tool(query: str, top_k: int = 5, source_file: str = "", file_type: str = "") -> str:
    """Retrieve evidence snippets from the local knowledge base for Agentic RAG answering."""
    filters: MetadataFilters = {}
    if source_file:
        filters["source_file"] = source_file
    if file_type:
        filters["file_type"] = file_type

    results = _default_rag_tools.retrieve(query=query, top_k=top_k, filters=filters or None)
    if not results:
        return "未检索到相关证据片段。"
    return _default_rag_tools.format_snippets(results)


def get_langchain_rag_tools() -> list[Any]:
    return [rag_retrieve_tool]
