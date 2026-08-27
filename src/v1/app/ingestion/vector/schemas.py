from dataclasses import dataclass

@dataclass
class Chunk:
  embedding: list[float]
  position: int
  content: str

@dataclass
class ProcessedDocument:
  metadata: dict[str, any]
  chunks: list[Chunk]
