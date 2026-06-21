from app.rag.schema import RetrievalResult


def render_retrieval_results(results: list[RetrievalResult]) -> list[dict]:
    return [
        {
            "chunk_id": item.chunk_id,
            "source_file": item.source_file,
            "score": item.score,
            "content": item.content,
        }
        for item in results
    ]
