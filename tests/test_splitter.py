from app.rag.schema import RawDocument
from app.rag.splitter import DocumentSplitter


def test_splitter_generates_unique_chunk_ids_for_same_source_file() -> None:
    splitter = DocumentSplitter()
    documents = [
        RawDocument(
            content="第一页内容",
            source_file="sample.pdf",
            file_type="pdf",
            metadata={"page": 1},
        ),
        RawDocument(
            content="第二页内容",
            source_file="sample.pdf",
            file_type="pdf",
            metadata={"page": 2},
        ),
    ]

    chunks = splitter.split(documents)
    chunk_ids = [chunk.chunk_id for chunk in chunks]

    assert chunk_ids == ["sample.pdf::chunk::0", "sample.pdf::chunk::1"]
    assert len(chunk_ids) == len(set(chunk_ids))
    assert chunks[0].metadata["page"] == 1
    assert chunks[1].metadata["page"] == 2
