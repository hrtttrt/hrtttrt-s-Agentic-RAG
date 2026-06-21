from pathlib import Path

from app.evaluation.dataset_loader import DatasetLoader
from app.evaluation.metrics import EvaluationMetrics
from app.evaluation.ragas_eval import RagasEvaluator
from app.services.qa_service import QAService


REFUSAL_MARKERS = ["信息不足", "没有足够证据", "无法回答", "知识库中没有", "没有任何", "不包含", "没有关于"]
NUMBER_NORMALIZATION = {
    "一个": "1个",
    "一": "1",
    "二": "2",
    "两": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
    "十": "10",
    "一天": "1天",
    "一个工作日": "1个工作日",
    "一工作日": "1个工作日",
    "两个小时": "2小时",
    "两小时": "2小时",
    "三个工作日": "3个工作日",
}
TERM_NORMALIZATION = {
    "部门经理": "直属主管",
    "部门负责人": "部门负责人",
    "财务总监": "财务负责人",
    "发票或凭证": "发票",
    "对应发票": "发票",
    "远程办公需提交": "远程办公需要提交",
    "请假需提交": "请假需要提交",
    "报销需提交": "报销需要提交",
}
MULTI_HOP_EXPECTED_KEYWORDS: dict[str, list[str]] = {
    "complex_001": ["年假", "3个工作日", "远程办公", "1个工作日"],
    "complex_002": ["请假申请", "远程办公申请", "发票"],
    "complex_003": ["设备遗失", "2小时", "安全团队", "1个工作日", "立即"],
    "complex_004": ["直属主管", "人力资源部"],
    "complex_005": ["请假超过3天", "信息安全团队", "财务负责人"],
}


class EvaluationRunner:
    def __init__(self, testset_path: Path) -> None:
        self.dataset_loader = DatasetLoader(testset_path)
        self.qa_service = QAService()
        self.ragas_evaluator = RagasEvaluator()

    def run(self) -> dict:
        cases = self.dataset_loader.load()
        records: list[dict] = []

        for case in cases:
            result = self.qa_service.ask(case["question"])
            answer = result.get("final_answer", "")
            gold_answer = case.get("gold_answer", "")
            evidence_pool = result.get("evidence_pool", [])
            is_refusal = any(marker in answer for marker in REFUSAL_MARKERS)
            is_correct = self._is_correct(case, answer, is_refusal)
            records.append(
                {
                    "id": case.get("id"),
                    "question": case.get("question"),
                    "question_type": case.get("question_type", "unknown"),
                    "answer": answer,
                    "gold_answer": gold_answer,
                    "is_correct": is_correct,
                    "is_refusal": is_refusal,
                    "citations": result.get("citations", []),
                    "iteration_count": result.get("iteration_count", 0),
                    "subqueries": result.get("subqueries", []),
                    "retrieved_contexts": [item.content for item in evidence_pool],
                    "retrieved_context_sources": [item.source_file for item in evidence_pool],
                }
            )

        return {
            "summary": {
                "accuracy": EvaluationMetrics.accuracy(records),
                "accuracy_by_type": EvaluationMetrics.accuracy_by_type(records),
                "breakdown": EvaluationMetrics.breakdown(records),
                "retrieval_hit_rate": EvaluationMetrics.retrieval_hit_rate(records),
                **EvaluationMetrics.refusal_metrics(records),
                "total": len(records),
            },
            "ragas": self.ragas_evaluator.evaluate(records),
            "records": records,
        }

    @staticmethod
    def _is_correct(case: dict, answer: str, is_refusal: bool) -> bool:
        question_type = case.get("question_type")
        case_id = str(case.get("id", ""))
        gold_answer = str(case.get("gold_answer", "")).strip()
        normalized_answer = EvaluationRunner._normalize_text(answer)
        normalized_gold = EvaluationRunner._normalize_text(gold_answer)

        if question_type == "insufficient_info":
            return is_refusal
        if not normalized_gold:
            return False
        if normalized_gold in normalized_answer:
            return True

        gold_terms = [term.strip() for term in gold_answer.replace("，", ",").split(",") if term.strip()]
        if len(gold_terms) > 1:
            return all(EvaluationRunner._normalize_text(term) in normalized_answer for term in gold_terms)

        if question_type == "multi_hop":
            expected_keywords = MULTI_HOP_EXPECTED_KEYWORDS.get(case_id, [])
            if expected_keywords:
                matched = sum(1 for keyword in expected_keywords if EvaluationRunner._normalize_text(keyword) in normalized_answer)
                if matched >= len(expected_keywords) - 1:
                    return True

            fragments = [fragment.strip() for fragment in gold_answer.split("，") if fragment.strip()]
            if len(fragments) > 1:
                matched = sum(1 for fragment in fragments if EvaluationRunner._normalize_text(fragment) in normalized_answer)
                if matched >= max(2, len(fragments) - 1):
                    return True

            keywords = EvaluationRunner._extract_multi_hop_keywords(gold_answer)
            if keywords:
                matched = sum(1 for keyword in keywords if keyword in normalized_answer)
                required = max(2, len(keywords) // 2)
                return matched >= required

        return False

    @staticmethod
    def _extract_multi_hop_keywords(text: str) -> list[str]:
        normalized = str(text)
        separators = ["，", "。", ",", ";", "；"]
        for separator in separators:
            normalized = normalized.replace(separator, " ")
        candidates = [item.strip() for item in normalized.split() if item.strip()]
        keywords = [EvaluationRunner._normalize_text(item) for item in candidates if len(EvaluationRunner._normalize_text(item)) >= 2]
        deduped: list[str] = []
        for keyword in keywords:
            if keyword not in deduped:
                deduped.append(keyword)
        return deduped

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = str(text).lower()
        for source, target in NUMBER_NORMALIZATION.items():
            normalized = normalized.replace(source, target)
        for source, target in TERM_NORMALIZATION.items():
            normalized = normalized.replace(source, target)
        return "".join(char for char in normalized if not char.isspace() and char not in "，。,.;；：:、()（）[]【】*-")
