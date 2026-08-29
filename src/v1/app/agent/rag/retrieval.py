from v1.infra.database import PostgresClient
from dataclasses import dataclass
from uuid import UUID

QUERY = """
SELECT id, document_id, content, position, embedding <=> %s AS distance
FROM chunks
ORDER BY embedding <=> %s
LIMIT %s
"""

@dataclass
class VectorSearch:
  embedding: list[float]
  top_k: int = 5

@dataclass
class QueryResult:
  id : UUID
  document_id: UUID
  content: str
  position: int
  distance: float

class Retrieval:
  def __init__(self, client: PostgresClient):
    self.client = client

  def vector_search(self, search:VectorSearch) -> list[QueryResult]:
    with self.client.connection() as conn:
      with conn.cursor() as cursor:
        result = cursor.execute(
          QUERY,
          (search.embedding, search.embedding, search.top_k)
        )

    return [QueryResult(*row) for row in result.fetchall()]
