from pathlib import Path

import pymupdf
import pytest
import typer
from langchain_core.documents import Document

from v1.app.ingestion.loader.pdf_loader import PdfLoader
from v1.main import document_from_path


def _write_pdf(path: Path, text: str) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()
    return path


def test_pdf_loader_extracts_text_and_names_the_file(tmp_path: Path):
    path = _write_pdf(tmp_path / "notes.pdf", "hello from pdf")
    document = PdfLoader().load(path)

    assert "hello from pdf" in document.page_content
    assert document.metadata["name"] == "notes.pdf"
    assert document.metadata["source"] == str(path)


def test_pdf_loader_joins_pages_when_loader_returns_many(tmp_path: Path, monkeypatch):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4")

    class FakeLoader:
        def __init__(self, file_path, **_kwargs):
            self.file_path = file_path

        def load(self):
            return [
                Document(page_content="page one", metadata={"page": 0}),
                Document(page_content="page two", metadata={"page": 1}),
            ]

    monkeypatch.setattr(
        "v1.app.ingestion.loader.pdf_loader.PyMuPDFLoader",
        FakeLoader,
    )
    document = PdfLoader().load(path)

    assert document.page_content == "page one\n\npage two"
    assert document.metadata["name"] == "doc.pdf"
    assert document.metadata["source"] == str(path)


def test_pdf_loader_rejects_empty_extract(tmp_path: Path, monkeypatch):
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"%PDF-1.4")

    class FakeLoader:
        def __init__(self, file_path, **_kwargs):
            self.file_path = file_path

        def load(self):
            return [Document(page_content="   ", metadata={})]

    monkeypatch.setattr(
        "v1.app.ingestion.loader.pdf_loader.PyMuPDFLoader",
        FakeLoader,
    )
    with pytest.raises(ValueError, match="No text extracted"):
        PdfLoader().load(path)


def test_document_from_path_loads_a_pdf(tmp_path: Path):
    path = _write_pdf(tmp_path / "notes.pdf", "wired")
    document = document_from_path(path)

    assert "wired" in document.page_content
    assert document.metadata["name"] == "notes.pdf"


def test_document_from_path_rejects_missing_file(tmp_path: Path):
    with pytest.raises(typer.BadParameter, match="Not a file"):
        document_from_path(tmp_path / "missing.pdf")


def test_document_from_path_rejects_non_pdf(tmp_path: Path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    with pytest.raises(typer.BadParameter, match=r"not a \.pdf file"):
        document_from_path(path)
