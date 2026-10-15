
import pymupdf
import pytest

from src.ingest import load_document


def test_load_txt_file(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("Hello RAG", encoding="utf-8")

    records = load_document(path)

    assert len(records) == 1
    assert records[0]["source"] == "sample.txt"
    assert records[0]["page"] is None
    assert records[0]["text"] == "Hello RAG"


def test_load_pdf_file(tmp_path):
    path = tmp_path / "sample.pdf"

    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "RAG PDF test")
        pdf.save(path)

    records = load_document(path)

    assert len(records) == 1
    assert records[0]["source"] == "sample.pdf"
    assert records[0]["page"] == 1
    assert "RAG PDF test" in records[0]["text"]


def test_missing_file_raises_error(tmp_path):
    path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        load_document(path)


def test_unsupported_extension_raises_error(tmp_path):
    path = tmp_path / "sample.docx"
    path.write_text("Example", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(path)


def test_empty_txt_returns_no_records(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")

    assert load_document(path) == []
