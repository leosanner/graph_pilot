from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

from v1.app.ingestion.loader.loader import Loader


class PdfLoader(Loader):
    def load(self, path: Path) -> Document:
        path = Path(path)
        pages = PyMuPDFLoader(str(path), mode="single").load()

        content = "\n\n".join(
            page.page_content for page in pages if page.page_content.strip()
        )
        if not content:
            raise ValueError(f"No text extracted from {path}")

        metadata = dict(pages[0].metadata) if pages else {}
        metadata["source"] = str(path)
        metadata["name"] = path.name

        return Document(page_content=content, metadata=metadata)
