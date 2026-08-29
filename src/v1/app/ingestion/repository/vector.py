from v1.infra.database import PostgresClient
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4
from psycopg.types.json import Jsonb


INSERT_VECTOR_QUERY = """
    INSERT INTO chunks (id, document_id, content, position, embedding, metadata)
    VALUES (%s, %s, %s, %s, %s, %s)
"""


INSERT_DOCUMENT_QUERY = """
    INSERT INTO documents (id, name, created_at, updated_at)
    VALUES (%s, %s, %s, %s)
"""


type Json = str | int | float | bool | None | list[Json] | dict[str, Json]

def utc_now() -> datetime:
  return datetime.now(UTC)

@dataclass
class ChunkEntity:
  document_id: UUID
  content: str
  position: int
  embedding: list[float]
  id: UUID = field(default_factory=uuid4)
  metadata: dict[str, Json] = field(default_factory=dict)

@dataclass
class DocumentEntity:
  name: str
  id: UUID = field(default_factory=uuid4)
  updated_at: datetime | None = None
  created_at: datetime = field(default_factory=utc_now)


class VectorRepository:
  def __init__(self, db_client: PostgresClient):
    self.client = db_client

  def insert_chunks(self, entities: list[ChunkEntity]) -> None:
    with self.client.connection() as conn:
      with conn.cursor() as cursor:
        cursor.executemany(
          INSERT_VECTOR_QUERY,
          [
            (
              entity.id,
              entity.document_id,
              entity.content,
              entity.position,
              entity.embedding,
              Jsonb(entity.metadata),
            )
            for entity in entities
          ]
        )

  def insert_document(self, document: DocumentEntity) -> None:
    with self.client.connection() as conn:
      with conn.cursor() as cursor:
        cursor.execute(
          INSERT_DOCUMENT_QUERY,
          (document.id, document.name, document.created_at, document.updated_at)
        )
