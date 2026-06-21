import json
from pathlib import Path


class DatasetLoader:
    def __init__(self, testset_path: Path) -> None:
        self.testset_path = testset_path

    def load(self) -> list[dict]:
        return json.loads(self.testset_path.read_text(encoding="utf-8"))
