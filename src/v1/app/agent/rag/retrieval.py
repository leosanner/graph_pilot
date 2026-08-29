from v1.infra.database import PostgresClient
from dataclasses import dataclass


@dataclass
class VectorSearch:
  embedding: list[float]
  top_k: int = 5


class Retrieval:
  def __init__(self, client: PostgresClient):
    self.client = client

  def vector_search(self, search:VectorSearch):
    pass
