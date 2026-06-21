import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app/ui/streamlit_app.py"],
        check=False,
        cwd=ROOT_DIR,
    )


if __name__ == "__main__":
    main()
