from abc import ABC, abstractmethod

class VectorRepository(ABC):

  @abstractmethod
  def search(self, query: str, sources:int = 5):
    ...

class GraphRepository(ABC):

  @abstractmethod
  def search(self):
    ...
