from unittest.mock import MagicMock
from uuid import uuid4

from psycopg.types.json import Jsonb

from v1.app.ingestion.repository.vector import ChunkEntity, VectorRepository


def test_insert_chunks_adapts_metadata_as_jsonb():
    client = MagicMock()
    conn = MagicMock()
    cursor = MagicMock()
    client.connection.return_value.__enter__.return_value = conn
    conn.cursor.return_value.__enter__.return_value = cursor

    entity = ChunkEntity(
        document_id=uuid4(),
        content="hello",
        position=0,
        embedding=[0.1, 0.2],
        metadata={"source": "notes.pdf", "chunks_count": 1},
    )
    VectorRepository(client).insert_chunks([entity])

    query, rows = cursor.executemany.call_args.args
    assert "INSERT INTO chunks" in query
    row = rows[0]
    assert isinstance(row[-1], Jsonb)
    assert row[-1].obj == entity.metadata
