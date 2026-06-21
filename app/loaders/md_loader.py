from pathlib import Path

from app.loaders.base import BaseLoader
from app.rag.schema import RawDocument


class MarkdownLoader(BaseLoader):
    supported_extensions = (".md",)

    def load(self, file_path: Path) -> list[RawDocument]:
        text = file_path.read_text(encoding="utf-8")
        return [RawDocument(content=text, source_file=file_path.name, file_type="md")]
