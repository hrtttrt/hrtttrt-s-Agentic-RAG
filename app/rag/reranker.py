import re
from dataclasses import replace

from app.rag.schema import RetrievalResult


class SimpleReranker:
    def rerank(self, query: str, docs: list[RetrievalResult]) -> list[RetrievalResult]:
        if not docs:
            return []

        query_ngrams = self._char_ngrams(query)
        rescored: list[tuple[float, RetrievalResult]] = []
        for doc in docs:
            semantic_score = 1.0 / (1.0 + max(doc.score, 0.0))
            lexical_score = self._lexical_overlap(query_ngrams, doc.content)
            phrase_bonus = self._phrase_bonus(query, doc.content)
            final_score = semantic_score * 0.55 + lexical_score * 0.45 + phrase_bonus
            rescored.append((final_score, replace(doc, score=final_score)))

        rescored.sort(key=lambda item: item[0], reverse=True)
        ranked_docs = [item[1] for item in rescored]

        filtered = [doc for doc in ranked_docs if doc.score >= 0.2]
        if filtered:
            return filtered
        return ranked_docs

    @staticmethod
    def _lexical_overlap(query_ngrams: set[str], content: str) -> float:
        if not query_ngrams:
            return 0.0
        content_ngrams = SimpleReranker._char_ngrams(content)
        if not content_ngrams:
            return 0.0
        return len(query_ngrams & content_ngrams) / len(query_ngrams)

    @staticmethod
    def _phrase_bonus(query: str, content: str) -> float:
        query_text = SimpleReranker._normalize_text(query)
        content_text = SimpleReranker._normalize_text(content)
        if not query_text or not content_text:
            return 0.0

        bonus = 0.0
        for phrase in SimpleReranker._focus_phrases(query):
            if phrase and phrase in content_text:
                bonus += 0.08
        return min(bonus, 0.24)

    @staticmethod
    def _focus_phrases(text: str) -> list[str]:
        normalized = SimpleReranker._normalize_text(text)
        if not normalized:
            return []

        stop_phrases = {
            "比较", "对比", "总结", "归纳", "分别", "说明", "哪些", "需要", "什么", "多久", "多少", "是谁", "为什么",
            "问题", "情况", "分别说明", "分别总结", "申请", "提交", "材料", "要求", "时限", "参与",
        }
        chunks = re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9]{2,}", normalized)
        return [chunk for chunk in chunks if chunk not in stop_phrases]

    @staticmethod
    def _char_ngrams(text: str, min_n: int = 2, max_n: int = 3) -> set[str]:
        normalized = SimpleReranker._normalize_text(text)
        if len(normalized) < min_n:
            return {normalized} if normalized else set()

        grams: set[str] = set()
        for size in range(min_n, max_n + 1):
            for idx in range(0, max(0, len(normalized) - size + 1)):
                grams.add(normalized[idx : idx + size])
        return grams

    @staticmethod
    def _normalize_text(text: str) -> str:
        lowered = text.lower()
        return re.sub(r"[^\u4e00-\u9fffa-z0-9]", "", lowered)
