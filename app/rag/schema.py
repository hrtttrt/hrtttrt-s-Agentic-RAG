from dataclasses import dataclass, field
from typing import Any


@dataclass
class RawDocument:
    content: str
    source_file: str
    file_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkDocument:
    chunk_id: str
    content: str
    source_file: str
    file_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    chunk_id: str
    content: str
    score: float
    source_file: str
    metadata: dict[str, Any] = field(default_factory=dict)
