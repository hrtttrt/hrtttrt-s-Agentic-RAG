from dataclasses import dataclass, field
from pathlib import Path

from app.loaders.router import LoaderRouter
from app.rag.embeddings import EmbeddingService
from app.rag.schema import RawDocument
from app.rag.splitter import DocumentSplitter
from app.rag.vector_store import ChromaVectorStore


@dataclass
class IngestReport:
    indexed_chunks: int = 0
    loaded_documents: int = 0
    skipped_files: list[dict[str, str]] = field(default_factory=list)


class IngestService:
    def __init__(self) -> None:
        self.loader_router = LoaderRouter()
        self.splitter = DocumentSplitter()
        self.embedding_service = EmbeddingService()
        self.vector_store = ChromaVectorStore()

    def ingest_directory(self, directory: Path, reset: bool = True) -> int:
        return self.ingest_directory_with_report(directory, reset=reset).indexed_chunks

    def ingest_directory_with_report(self, directory: Path, reset: bool = True) -> IngestReport:
        raw_documents: list[RawDocument] = []
        skipped_files: list[dict[str, str]] = []

        for file_path in directory.rglob("*"):
            if not file_path.is_file():
                continue
            try:
                raw_documents.extend(self.loader_router.load(file_path))
            except Exception as exc:
                skipped_files.append({"path": str(file_path), "reason": str(exc)})

        chunks = self.splitter.split(raw_documents)
        if not chunks:
            return IngestReport(
                indexed_chunks=0,
                loaded_documents=len(raw_documents),
                skipped_files=skipped_files,
            )

        if reset:
            self.vector_store.reset_collection()

        embeddings = self.embedding_service.embed_documents([chunk.content for chunk in chunks])
        self.vector_store.add_chunks(chunks, embeddings)
        return IngestReport(
            indexed_chunks=len(chunks),
            loaded_documents=len(raw_documents),
            skipped_files=skipped_files,
        )

    def ingest_files(self, paths: list[Path], reset: bool = False) -> IngestReport:
        raw_documents: list[RawDocument] = []
        skipped_files: list[dict[str, str]] = []

        for file_path in paths:
            try:
                raw_documents.extend(self.loader_router.load(file_path))
            except Exception as exc:
                skipped_files.append({"path": str(file_path), "reason": str(exc)})

        chunks = self.splitter.split(raw_documents)
        if chunks:
            if reset:
                self.vector_store.reset_collection()
            embeddings = self.embedding_service.embed_documents([chunk.content for chunk in chunks])
            self.vector_store.add_chunks(chunks, embeddings)

        return IngestReport(
            indexed_chunks=len(chunks),
            loaded_documents=len(raw_documents),
            skipped_files=skipped_files,
        )
