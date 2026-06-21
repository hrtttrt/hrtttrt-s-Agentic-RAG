import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]
STREAMLIT_SECRETS_PATH = BASE_DIR / ".streamlit" / "secrets.toml"


def _load_streamlit_secrets() -> None:
    if not STREAMLIT_SECRETS_PATH.exists():
        return

    try:
        import tomllib

        secret_values = tomllib.loads(STREAMLIT_SECRETS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return

    for key, value in secret_values.items():
        env_key = str(key).upper()
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                os.environ.setdefault(str(nested_key).upper(), str(nested_value))
            continue
        os.environ.setdefault(env_key, str(value))


_load_streamlit_secrets()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Agentic RAG"
    app_env: Literal["dev", "test", "prod"] = "dev"
    debug: bool = True

    llm_provider: str = "openai_compatible"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model_name: str = "gpt-4o-mini"
    llm_temperature: float = 0.1

    embedding_backend: Literal["sentence_transformers", "hash", "openai_compatible"] = "sentence_transformers"
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_dimension: int = 384
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"

    vector_db_path: Path = Field(default=BASE_DIR / "chroma_db")
    knowledge_base_dir: Path = Field(default=BASE_DIR / "data" / "knowledge_base" / "raw")
    processed_data_dir: Path = Field(default=BASE_DIR / "data" / "knowledge_base" / "processed")
    reports_dir: Path = Field(default=BASE_DIR / "reports")

    chunk_size: int = 600
    chunk_overlap: int = 120
    top_k: int = 5
    max_agent_iterations: int = 3
    enable_ragas_eval: bool = True

    def ensure_runtime_directories(self) -> None:
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        self.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_runtime_directories()
