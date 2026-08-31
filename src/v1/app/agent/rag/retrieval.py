from dataclasses import dataclass
from uuid import UUID

from pgvector import Vector

from v1.infra.database import PostgresClient

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
    id: UUID
    document_id: UUID
    content: str
    position: int
    distance: float


class Retrieval:
    def __init__(self, client: PostgresClient):
        self.client = client

    def vector_search(self, search: VectorSearch) -> list[QueryResult]:
        # pgvector only registers dumpers for Vector and ndarray, so a plain list
        # would bind as float8[] and miss the <=> operator.
        embedding = Vector(search.embedding)

        with self.client.connection() as conn, conn.cursor() as cursor:
            cursor.execute(QUERY, (embedding, embedding, search.top_k))
            rows = cursor.fetchall()

        return [QueryResult(*row) for row in rows]
