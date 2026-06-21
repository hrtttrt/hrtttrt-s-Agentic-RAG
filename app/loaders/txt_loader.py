from pathlib import Path

from app.loaders.base import BaseLoader
from app.rag.schema import RawDocument


class TxtLoader(BaseLoader):
    supported_extensions = (".txt",)

    def load(self, file_path: Path) -> list[RawDocument]:
        text = file_path.read_text(encoding="utf-8")
        return [RawDocument(content=text, source_file=file_path.name, file_type="txt")]
