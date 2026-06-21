from collections import Counter
from typing import Iterable


class EvaluationMetrics:
    @staticmethod
    def accuracy(records: Iterable[dict]) -> float:
        items = list(records)
        if not items:
            return 0.0
        correct = sum(1 for item in items if item.get("is_correct"))
        return correct / len(items)

    @staticmethod
    def breakdown(records: Iterable[dict]) -> dict[str, int]:
        items = list(records)
        counter = Counter(item.get("question_type", "unknown") for item in items)
        return dict(counter)

    @staticmethod
    def accuracy_by_type(records: Iterable[dict]) -> dict[str, float]:
        items = list(records)
        grouped: dict[str, list[dict]] = {}
        for item in items:
            grouped.setdefault(item.get("question_type", "unknown"), []).append(item)
        return {question_type: EvaluationMetrics.accuracy(group) for question_type, group in grouped.items()}

    @staticmethod
    def refusal_metrics(records: Iterable[dict]) -> dict[str, float]:
        items = list(records)
        true_refusals = [item for item in items if item.get("question_type") == "insufficient_info"]
        predicted_refusals = [item for item in items if item.get("is_refusal")]
        correct_refusals = [item for item in predicted_refusals if item.get("question_type") == "insufficient_info"]

        precision = len(correct_refusals) / len(predicted_refusals) if predicted_refusals else 0.0
        recall = len(correct_refusals) / len(true_refusals) if true_refusals else 0.0
        return {"refusal_precision": precision, "refusal_recall": recall}

    @staticmethod
    def retrieval_hit_rate(records: Iterable[dict]) -> float:
        items = list(records)
        answerable = [item for item in items if item.get("question_type") != "insufficient_info"]
        if not answerable:
            return 0.0
        hits = sum(1 for item in answerable if item.get("citations"))
        return hits / len(answerable)
