from app.rag.schema import RetrievalResult
from app.rag.tools import RAGTools


def test_format_snippets() -> None:
    results = [
        RetrievalResult(chunk_id="1", content="内容A", score=0.1, source_file="a.txt"),
        RetrievalResult(chunk_id="2", content="内容B", score=0.2, source_file="b.txt"),
    ]

    text = RAGTools.format_snippets(results)
    assert "来源: a.txt" in text
    assert "内容B" in text
