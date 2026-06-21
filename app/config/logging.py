from pathlib import Path
import sys

from loguru import logger


LOG_DIR = Path(__file__).resolve().parents[2] / "reports" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOG_DIR / "app.log", level="INFO", rotation="10 MB", encoding="utf-8")
