from abc import ABC, abstractmethod
from pathlib import Path
from langchain_core.documents import Document

# Factory
class Loader(ABC):

  @abstractmethod
  def load(path: Path) -> Document:
    ...
