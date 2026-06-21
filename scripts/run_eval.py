import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config.settings import settings
from app.services.eval_service import EvalService


DEFAULT_TESTSET = ROOT_DIR / "data" / "testsets" / "cases.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Agentic RAG evaluation.")
    parser.add_argument(
        "--testset",
        type=Path,
        default=DEFAULT_TESTSET,
        help="Path to the evaluation testset JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the full evaluation report as JSON.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the evaluation report file automatically.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the full evaluation result JSON to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    testset_path = resolve_testset_path(args.testset)

    if not testset_path.exists():
        raise FileNotFoundError(f"Testset file not found: {testset_path}")

    service = EvalService(testset_path)
    result = service.run()

    if args.pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if not args.no_save:
        output_path = resolve_output_path(args.output)
        write_report(output_path, result)
        print(f"{output_path}")


def resolve_testset_path(testset_path: Path) -> Path:
    if testset_path.is_absolute():
        return testset_path
    return (ROOT_DIR / testset_path).resolve()


def resolve_output_path(output_path: Path | None) -> Path:
    if output_path is not None:
        if output_path.is_absolute():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return output_path
        final_path = (ROOT_DIR / output_path).resolve()
        final_path.parent.mkdir(parents=True, exist_ok=True)
        return final_path

    reports_dir = settings.reports_dir / "evaluations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return reports_dir / f"eval_report_{timestamp}.json"


def write_report(output_path: Path, result: dict[str, Any]) -> None:
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
