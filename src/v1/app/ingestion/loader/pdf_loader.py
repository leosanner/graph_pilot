from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from pathlib import Path

class PdfLoader:
  def __init__(self):
    pass

  def load(self, path:Path) -> Document:
    return PyMuPDFLoader(path).load()
