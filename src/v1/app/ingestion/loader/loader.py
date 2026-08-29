from abc import ABC, abstractmethod
from pathlib import Path
from langchain_core.documents import Document

class Loader(ABC):

  @abstractmethod
  def load(self, path: Path) -> Document:
    ...
