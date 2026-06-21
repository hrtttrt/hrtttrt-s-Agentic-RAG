from pathlib import Path

from app.loaders.txt_loader import TxtLoader


def test_txt_loader(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello world", encoding="utf-8")

    loader = TxtLoader()
    documents = loader.load(file_path)

    assert len(documents) == 1
    assert documents[0].content == "hello world"
