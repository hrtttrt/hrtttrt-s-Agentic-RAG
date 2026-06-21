from pathlib import Path

import fitz

from app.loaders.base import BaseLoader
from app.rag.schema import RawDocument


class PDFLoader(BaseLoader):
    supported_extensions = (".pdf",)

    def load(self, file_path: Path) -> list[RawDocument]:
        documents: list[RawDocument] = []
        pdf = fitz.open(file_path)
        for page_idx, page in enumerate(pdf, start=1):
            text = page.get_text().strip()
            if text:
                documents.append(
                    RawDocument(
                        content=text,
                        source_file=file_path.name,
                        file_type="pdf",
                        metadata={"page": page_idx},
                    )
                )
        return documents
