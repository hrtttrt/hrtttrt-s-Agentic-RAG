from pathlib import Path

from app.config.logging import logger
from app.config.settings import settings
from app.services.ingest_service import IngestService
from app.services.qa_service import QAService


def main() -> None:
    logger.info("Starting {}", settings.app_name)

    kb_path = Path(settings.knowledge_base_dir)
    ingest_service = IngestService()
    indexed_chunks = ingest_service.ingest_directory(kb_path)
    logger.info("Indexed {} chunks from {}", indexed_chunks, kb_path)

    qa_service = QAService()

    while True:
        query = input("\n请输入问题（输入 exit 退出）：").strip()
        if query.lower() in {"exit", "quit"}:
            break

        result = qa_service.ask(query)
        print("\n回答：")
        print(result.get("final_answer", ""))
        print("\n引用：", result.get("citations", []))


if __name__ == "__main__":
    main()
