from app.evaluation.metrics import EvaluationMetrics


def test_accuracy() -> None:
    records = [{"is_correct": True}, {"is_correct": False}, {"is_correct": True}]
    assert EvaluationMetrics.accuracy(records) == 2 / 3
