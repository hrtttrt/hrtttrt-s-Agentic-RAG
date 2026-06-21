from abc import ABC, abstractmethod
from pathlib import Path

from app.rag.schema import RawDocument


class BaseLoader(ABC):
    supported_extensions: tuple[str, ...] = tuple()

    @abstractmethod
    def load(self, file_path: Path) -> list[RawDocument]:
        raise NotImplementedError
