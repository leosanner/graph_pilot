from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pgvector import Vector

from v1.app.agent.rag.retrieval import Retrieval, VectorSearch


def _client(rows: list[tuple]) -> tuple[MagicMock, MagicMock]:
    client = MagicMock()
    conn = MagicMock()
    cursor = MagicMock()
    client.connection.return_value.__enter__.return_value = conn
    conn.cursor.return_value.__enter__.return_value = cursor
    cursor.fetchall.return_value = rows
    return client, cursor


def test_vector_search_reads_rows_before_the_cursor_closes():
    chunk_id, document_id = uuid4(), uuid4()
    client, cursor = _client([(chunk_id, document_id, "a passage", 3, 0.25)])

    results = Retrieval(client).vector_search(
        VectorSearch(embedding=[0.1, 0.2, 0.3], top_k=2)
    )

    cursor.fetchall.assert_called_once()
    assert cursor.fetchall.call_count == 1
    assert len(results) == 1
    assert results[0].id == chunk_id
    assert results[0].document_id == document_id
    assert results[0].content == "a passage"
    assert results[0].position == 3
    assert results[0].distance == 0.25


def test_vector_search_binds_the_embedding_as_a_vector():
    client, cursor = _client([])

    Retrieval(client).vector_search(VectorSearch(embedding=[0.1, 0.2], top_k=7))

    _query, params = cursor.execute.call_args.args
    assert isinstance(params[0], Vector)
    assert params[0].to_list() == pytest.approx([0.1, 0.2])
    assert params[1] is params[0]
    assert params[2] == 7
