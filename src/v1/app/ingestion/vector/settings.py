from dataclasses import dataclass
from langchain_core.embeddings import Embeddings

@dataclass
class VectorSettings:
  chunk_size: int
  chunk_overlap: int
  model: Embeddings

@dataclass
class RuntimeSettings:
  batch_size: int = 32
  max_concurrent_batches: int = 4
  request_timeout: float = 60.0
  max_attempts: int = 3
  retry_base_delay: float = 0.5
