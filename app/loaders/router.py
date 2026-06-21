from pathlib import Path

from app.loaders.base import BaseLoader
from app.loaders.doc_loader import DocLoader
from app.loaders.md_loader import MarkdownLoader
from app.loaders.pdf_loader import PDFLoader
from app.loaders.ppt_loader import PPTLoader
from app.loaders.txt_loader import TxtLoader
from app.loaders.xlsx_loader import XLSXLoader


class LoaderRouter:
    def __init__(self) -> None:
        self.loaders: list[BaseLoader] = [
            TxtLoader(),
            MarkdownLoader(),
            PDFLoader(),
            DocLoader(),
            XLSXLoader(),
            PPTLoader(),
        ]

    def get_loader(self, file_path: Path) -> BaseLoader:
        suffix = file_path.suffix.lower()
        for loader in self.loaders:
            if suffix in loader.supported_extensions:
                return loader
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    def load(self, file_path: Path):
        loader = self.get_loader(file_path)
        return loader.load(file_path)
