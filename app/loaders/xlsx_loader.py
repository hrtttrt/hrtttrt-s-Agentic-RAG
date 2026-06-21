from pathlib import Path

import pandas as pd

from app.loaders.base import BaseLoader
from app.rag.schema import RawDocument


class XLSXLoader(BaseLoader):
    supported_extensions = (".xlsx",)

    def load(self, file_path: Path) -> list[RawDocument]:
        documents: list[RawDocument] = []
        excel_file = pd.ExcelFile(file_path)
        for sheet_name in excel_file.sheet_names:
            df = excel_file.parse(sheet_name)
            text = df.fillna("").to_csv(index=False)
            documents.append(
                RawDocument(
                    content=text,
                    source_file=file_path.name,
                    file_type="xlsx",
                    metadata={"sheet": sheet_name},
                )
            )
        return documents
