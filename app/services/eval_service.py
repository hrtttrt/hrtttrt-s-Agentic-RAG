from pathlib import Path

from app.evaluation.runner import EvaluationRunner


class EvalService:
    def __init__(self, testset_path: Path) -> None:
        self.runner = EvaluationRunner(testset_path=testset_path)

    def run(self) -> dict:
        return self.runner.run()
