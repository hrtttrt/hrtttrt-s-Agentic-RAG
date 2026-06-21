from collections import defaultdict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.settings import settings
from app.rag.schema import ChunkDocument, RawDocument


class DocumentSplitter:
    def __init__(self) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def split(self, documents: list[RawDocument]) -> list[ChunkDocument]:
        chunks: list[ChunkDocument] = []
        chunk_indexes_by_source: dict[str, int] = defaultdict(int)

        for doc_index, doc in enumerate(documents):
            parts = self.splitter.split_text(doc.content)
            for local_chunk_index, part in enumerate(parts):
                source_chunk_index = chunk_indexes_by_source[doc.source_file]
                chunk_indexes_by_source[doc.source_file] += 1
                metadata = {
                    **doc.metadata,
                    "source_document_index": doc_index,
                    "local_chunk_index": local_chunk_index,
                }
                chunks.append(
                    ChunkDocument(
                        chunk_id=f"{doc.source_file}::chunk::{source_chunk_index}",
                        content=part,
                        source_file=doc.source_file,
                        file_type=doc.file_type,
                        metadata=metadata,
                    )
                )
        return chunks
