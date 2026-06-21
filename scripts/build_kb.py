from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config.settings import settings
from app.services.ingest_service import IngestService


def main() -> None:
    service = IngestService()
    count = service.ingest_directory(Path(settings.knowledge_base_dir), reset=True)
    print(f"Indexed {count} chunks from knowledge base: {settings.knowledge_base_dir}")


if __name__ == "__main__":
    main()
